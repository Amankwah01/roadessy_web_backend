import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide main window

    file_paths = filedialog.askopenfilenames(
        title="Select video files (in order or they will be sorted)",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
    )

    return list(file_paths)

def create_file_list(files, output_txt="file_list.txt"):
    with open(output_txt, "w", encoding="utf-8") as f:
        for file in files:
            f.write(f"file '{file}'\n")
    return output_txt

def stitch_videos(file_list_txt, output_file, reencode=False):
    if reencode:
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", file_list_txt,
            "-c:v", "libx264",
            "-c:a", "aac",
            output_file
        ]
    else:
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", file_list_txt,
            "-c", "copy",
            output_file
        ]

    subprocess.run(cmd)

def main():
    print("Select your video clips...")
    files = select_files()

    if not files:
        print("No files selected.")
        return

    # Sort files by name (important for dashcam clips)
    files.sort()

    print("\nSelected files:")
    for f in files:
        print(f)

    # Ask user for output file
    output_file = filedialog.asksaveasfilename(
        title="Save output video as",
        defaultextension=".mp4",
        filetypes=[("MP4 file", "*.mp4")]
    )

    if not output_file:
        print("No output file selected.")
        return

    # Ask mode
    mode = input("\nChoose mode:\n1 = Fast (no re-encode)\n2 = Safe (re-encode)\nEnter 1 or 2: ")

    reencode = True if mode == "2" else False

    file_list_txt = create_file_list(files)

    print("\nProcessing... please wait")
    stitch_videos(file_list_txt, output_file, reencode)

    os.remove(file_list_txt)

    print("\nDone! Output saved to:", output_file)
    messagebox.showinfo("Success", f"Video saved to:\n{output_file}")

if __name__ == "__main__":
    main()