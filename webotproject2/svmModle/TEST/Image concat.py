from PIL import Image
import os

# Folder path
base_dir = r"E:\webot_project\webotproject2\svmModle\TEST"

# Two images to concatenate
img1_path = os.path.join(base_dir, "task_no_led.png")
img2_path = os.path.join(base_dir, "task_with_led.png")

# Output path
output_path = os.path.join(base_dir, "svm_result_concat.png")

# Load images
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# Normalize height (avoid misalignment if heights differ)
h = max(img1.height, img2.height)
img1 = img1.resize((img1.width, h))
img2 = img2.resize((img2.width, h))

# Create new canvas
new_img = Image.new("RGB", (img1.width + img2.width, h))

# Concatenate left–right
new_img.paste(img1, (0, 0))
new_img.paste(img2, (img1.width, 0))

# Save output
new_img.save(output_path)

print("✓ Concatenation complete. Output file:", output_path)
