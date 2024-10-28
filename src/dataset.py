import os
import requests

def download_images(image_urls, dataset_folder_path, trigger_word):
    # Ensure the dataset folder exists
    os.makedirs(dataset_folder_path, exist_ok=True)
    
    for index, url in enumerate(image_urls):
        try:
            # Fetch the image
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses

            # Determine the image extension
            image_extension = url.split('.')[-1]  # Get the extension from the URL
            image_name = f"{trigger_word} ({index + 1}).{image_extension}"  # Create the new image name
            
            # Define the complete file path
            file_path = os.path.join(dataset_folder_path, image_name)

            # Write the image to a file
            with open(file_path, 'wb') as f:
                f.write(response.content)

            print(f"Downloaded: {file_path}")  # Log the downloaded file
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")  # Log any errors





