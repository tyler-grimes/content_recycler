import yt_dlp
import whisper
import os
import pypdf


class AudioProcessor:
    """Takes a youtube link and transcribes it"""

    def __init__(self, output_dir="downloads"):
        self.output_dir = output_dir
        self.last_downloaded_file = None
        self.model = whisper.load_model("base")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def download_audio(self, url):
        """Download audio and capture the file path"""
        downloaded_file = None

        def progress_hook(d):
            nonlocal downloaded_file
            if d["status"] == "finished":
                # This gives us the path before conversion
                downloaded_file = d["filename"]

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": os.path.join(self.output_dir, "%(title)s.%(ext)s"),
            "progress_hooks": [progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

            # Convert the webm path to mp3 path
            if downloaded_file:
                # The downloaded file was .webm, but after processing it's .mp3
                self.last_downloaded_file = (
                    os.path.splitext(downloaded_file)[0] + ".mp3"
                )
                print(f"Audio file saved to: {self.last_downloaded_file}")
                return self.last_downloaded_file

        return None

    def transcribe_video(self):
        """Transcribe the downloaded audio file"""
        if not self.last_downloaded_file or not os.path.exists(
            self.last_downloaded_file
        ):
            raise FileNotFoundError(
                f"Audio file not found: {self.last_downloaded_file}"
            )

        print(f"Transcribing: {self.last_downloaded_file}")
        result = self.model.transcribe(self.last_downloaded_file)
        return result

    def process(self, url):
        """Download and transcribe a video"""
        # Download the audio
        audio_path = self.download_audio(url)
        if not audio_path:
            raise RuntimeError("Failed to download audio")

        # Transcribe it
        transcript = self.transcribe_video()

        # Return both the transcript text and the file path
        return transcript["text"]


def read_pdf(pdf: str):
    reader = pypdf.PdfReader(pdf)
    num_pages = len(reader.pages)

    text_of_pdf = ""

    for i in range(num_pages):
        page = reader.pages[i]
        text = page.extract_text()
        text_of_pdf += text

    return text_of_pdf
