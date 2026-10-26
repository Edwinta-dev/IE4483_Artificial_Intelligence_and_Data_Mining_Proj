import argparse
from pathlib import Path
from PIL import Image, ImageOps
from collections import Counter

# Repo root is two levels above src/ie4483/. Raw handoff goes in raw-data/datasets
# (the folder holding train/, val/, test/); output lands in data/, the pipeline's default root.
repo_root = Path(__file__).resolve().parents[2]
input_dir = repo_root / "raw-data" / "datasets"
output_dir = repo_root / "data"

image_size = (224, 224)  # Resize images to 224x224 pixels

supported_formats = (".jpg", ".jpeg", ".png")

def preprocess_images(input_dir, output_dir, image_size):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()

    if(input_resolved == output_resolved or input_resolved in output_resolved.parents):
        raise ValueError("Output directory cannot be the same as or a subdirectory of the input directory.")

    image_files = [
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_formats
    ]
    
    if not image_files:
        print("No supported image files found.")
        return

    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Images found: {len(image_files)}")
    print(f"Target size: {image_size[0]}x{image_size[1]}")
    print("Processing images...")

    processed_count = 0
    failed_list = []
    folder_count = Counter()

    for index, image_path in enumerate(image_files, start=1):
        try:
            relative_path = image_path.relative_to(input_dir)

            output_path = (output_dir / relative_path).with_suffix(".jpg")  # Ensure output is saved as .jpg
            
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(image_path) as original:
                # Convert to RGB if the image is in a different mode
                image = ImageOps.exif_transpose(original).convert("RGB") if original.mode != "RGB" else ImageOps.exif_transpose(original)
                
                # Resize the image
                image = ImageOps.fit(image, image_size, method=Image.Resampling.LANCZOS)

                image.save(output_path, format="JPEG", quality=95)

            processed_count += 1
            folder_count[relative_path.parent] += 1

            if index % 100 == 0 or index == len(image_files):
                print(f"Processed {index}/{len(image_files)} images...")

        except Exception as error:
            failed_list.append((str(image_path), str(error)))
            print (f"Failed to process {image_path} -- {error}")

    print(f"\n======== Summary ========")
    print(f"Total images found: {len(image_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed images: {failed_list}")
    print(f"Output folder: {output_dir}")

    if failed_list:
        print("\nFailed image details:")
        for path, error in failed_list:
            print(f"{path} -- {error}")

    print("\nPreprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize the raw dataset to 224x224 JPEGs.")
    parser.add_argument("--input", type=Path, default=input_dir)
    parser.add_argument("--output", type=Path, default=output_dir)
    args = parser.parse_args()
    preprocess_images(args.input, args.output, image_size)