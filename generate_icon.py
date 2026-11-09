from PIL import Image, ImageDraw, ImageFont
import os

def create_app_icon(output_path):
    size = (256, 256)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded rectangle background with gradient effect
    # Base background: deep slate #14141e
    for i in range(256):
        ratio = i / 255.0
        r = int(24 + (124 - 24) * ratio * 0.15)
        g = int(24 + (108 - 24) * ratio * 0.15)
        b = int(36 + (240 - 36) * ratio * 0.3)

    draw.rounded_rectangle([12, 12, 244, 244], radius=54, fill=(26, 26, 38, 255), outline=(124, 108, 240, 255), width=6)

    # Inner subtle glow
    draw.rounded_rectangle([20, 20, 236, 236], radius=46, outline=(124, 108, 240, 100), width=2)

    # Glowing robot head / code symbol
    # Code brackets < / > in cyan & violet
    # Left bracket <
    draw.line([(85, 95), (55, 128), (85, 161)], fill=(56, 189, 248, 255), width=14, joint="curve")
    # Right bracket >
    draw.line([(171, 95), (201, 128), (171, 161)], fill=(244, 114, 182, 255), width=14, joint="curve")

    # Center AI sparkle / bot face
    # Center circle/node
    draw.ellipse([108, 108, 148, 148], fill=(124, 108, 240, 255))
    draw.ellipse([118, 118, 138, 138], fill=(255, 255, 255, 255))

    # Antenna / crown
    draw.line([(128, 70), (128, 108)], fill=(124, 108, 240, 255), width=8)
    draw.ellipse([120, 60, 136, 76], fill=(56, 189, 248, 255))

    # Bottom connection dots
    draw.ellipse([88, 182, 104, 198], fill=(56, 189, 248, 255))
    draw.ellipse([120, 182, 136, 198], fill=(124, 108, 240, 255))
    draw.ellipse([152, 182, 168, 198], fill=(244, 114, 182, 255))

    # Save as .ico containing multiple sizes
    img.save(
        output_path,
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    )
    print(f"Generated icon at {output_path}")

if __name__ == "__main__":
    out_dir = r"C:\Users\Gurnoor\Desktop\gemma-chatbot"
    create_app_icon(os.path.join(out_dir, "app_icon.ico"))
