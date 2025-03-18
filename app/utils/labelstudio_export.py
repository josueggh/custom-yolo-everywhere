import os
import io
import json
import zipfile
from urllib.parse import quote
import requests

def export_annotations(project_id, api_key, export_format="YOLO_OBB_WITH_IMAGES"):
    """
    Export annotations from Label Studio using the specified export format.
    If the response is a ZIP archive (as is the case with YOLO_OBB_WITH_IMAGES),
    it extracts the contents to a directory and returns the directory path.
    """
    base_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
    encoded_format = quote(export_format)
    url = f"{base_url}/api/projects/{project_id}/export?exportType={encoded_format}"
    headers = {"Authorization": f"Token {api_key}"}
    response = requests.get(url, headers=headers)

    print(f"Response status: {response.status_code}")
    snippet = response.text[:200]
    print("Response text snippet:", snippet)

    if not response.ok:
        raise Exception(
            f"Failed to export annotations for project {project_id}: {response.status_code} {response.text}"
        )

    # Check if the response is a ZIP file (ZIP files start with "PK")
    if response.content.startswith(b"PK"):
        extract_dir = f"export_project_{project_id}"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_dir)
        print(f"Exported files extracted to: {extract_dir}")
        return extract_dir
    try:
        return response.json()
    except json.decoder.JSONDecodeError as e:
        raise Exception(f"Error decoding JSON for project {project_id}: {e}\nResponse text: {response.text}")