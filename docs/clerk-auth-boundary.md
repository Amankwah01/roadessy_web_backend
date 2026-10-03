# Clerk: where the boundary sits, and when the Backend API becomes necessary

## What the frontend is

Verified against the actual keys, not assumed:

- Instance: `guided-guppy-87.clerk.accounts.dev`
  (decoded from `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`; the returned JWKS `kid`
  prefix `ins_3AFtSBgTHPfr` matches the `ins_...` id that
  `GET /v1/instance` returns, so the two agree)
- `@clerk/nextjs` 6.39.7, Next.js 16.3.2
- Holds the session. `proxy.ts` protects `/dashboard(.*)` with
  `auth.protect()`.
- Mints the token the backend verifies, via `getToken()` in
  `lib/server/backend-token.ts` (prefers the `backend` token template, falls
  back to a plain session token).
- Proxies organization data to this service: `POST /api/me/organisations/sync`
  carries Clerk's organizations in the request body.

## What this service is

An OIDC resource server. It verifies a bearer token and trusts the claims.

- `OIDC_ISSUER` / `OIDC_JWKS_URL` point at the instance above.
- `security.py` fetches the JWKS, verifies RS256 signature, expiry and issuer,
  and rejects a token signed by any other key (checked with a forged token).
- **It never calls Clerk's API.** There is no secret key in the backend.

## Is the Backend API needed here?

Not for what this service does today, and adding it would cost more than it
returns:

1. **Identity already arrives verified.** `sub` and `org_id` come from the
   token's claims after signature and issuer checks. A BAPI round trip would
   re-derive what the signature already established.
2. **Organizations are already mirrored.** `Organisation`,
   `OrganisationMembership` and `UserProfile` are the source of truth for
   membership and permissions. BAPI would introduce a second source of truth
   and a reconciliation problem that does not exist now.
3. **It widens the blast radius.** Today a stolen backend cannot impersonate
   anyone, because it holds no credential that can mint or read user data. Adding
   `CLERK_SECRET_KEY` means it can. For an analytics service that is the wrong
   trade.
4. **It duplicates the frontend.** The organizations are already proxied through
   the Next server, which holds the session.

## The one case that would justify it

`POST /api/me/organisations/sync` trusts the caller about *which organizations
they belong to*. The body carries `external_organizations`, and the loop trusts
`id`, `name` and `slug` without checking any of them against the provider. The
caller's identity is verified; their **membership claim is not**.

The impact is bounded — it cannot read another user's data, and `_require_member`
still gates access to the caller's own rows — but it is the one place where a
client can assert something the backend has not verified. If you want that
closed, there are two options:

- **Fetch from Clerk instead of accepting the body.** The backend verifies each
  `external_org_id` against the Backend API and drops anything the caller is not
  actually a member of. Correct, and it is the reason to add a BAPI client.
- **Scope it to the verified claims.** If the token carries `org_id`, intersect
  the body's list with it and ignore the rest. No new credential, closes most of
  the gap, but only works if the token template emits `org_id`.

Note that Clerk's BAPI returns `403 error code: 1010` to Python's default
`urllib` User-Agent — that is Cloudflare blocking the client signature, not an
auth failure. A client must send a real `User-Agent` or every call fails in a
way that looks like a permissions problem.

## Versions

| Component | Version | Note |
|---|---|---|
| `@clerk/nextjs` | 6.39.7 | `^` in the frontend `package.json` |
| Next.js | 16.3.2 | `proxy.ts` is the Next 16 name for `middleware.ts` |
| Clerk instance | development | `environment_type: development` |
| JWKS algorithm | RS256 | single key |

## Verifying a change

`check_oidc_configuration()` logs at startup when `OIDC_*` is incomplete or when
leftover `CLERK_*` provider variables are present and dead. After changing the
issuer, confirm the boot log is clean and one authenticated request returns 200
rather than 503.