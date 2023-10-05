import stone
from json import dumps

def get_skin_tones(image_paths, image_type="color", palette=None, *other_args):
    skin_tone_results = {}
    
    for image_path in image_paths:
        # Process the image
        result = stone.process(image_path, image_type, palette, *other_args, return_report_image=False)
        
        # Assuming the result contains skin tone information, store it
        skin_tone_results[image_path] = result

    return skin_tone_results
def extract_info(data):
    info = {}
    
    # Extracting image details
    info["basename"] = data.get("basename")
    info["extension"] = data.get("extension")
    info["image_type"] = data.get("image_type")
    
    # Extracting face details
    face_data = data.get("faces", [{}])[0]  # Assuming there's at least one face
    
    # Dominant skin colors and percentages
    dominant_colors = face_data.get("dominant_colors", [])
    info["dominant_colors"] = {color_data["color"]: color_data["percent"] for color_data in dominant_colors}
    
    # Other face details
    info["skin_tone"] = face_data.get("skin_tone")
    info["tone_label"] = face_data.get("tone_label")
    info["accuracy"] = face_data.get("accuracy")
    
    return info
# Example usage:
images = ["../testcases/man_headup.jpg"]
results = get_skin_tones(images)
for image_path, result in results.items():
    result_info = extract_info(result)
    print(f"Image: {image_path}")
    print(result_info)
# Print or further process the results
# for image, result in results.items():
#     print(f"Image: {image} - Result: {result}")
