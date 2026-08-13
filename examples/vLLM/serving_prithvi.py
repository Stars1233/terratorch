import base64
from pathlib import Path

import requests

# Requirements :
# - install TerraTorch with the vLLM extra dependencies:
#   pip install terratorch[vllm]
# - start vllm in serving mode with the below args
# vllm serve ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
#   --skip-tokenizer-init --enforce-eager
#   --io-processor-plugin terratorch_segmentation
#   --enable-mm-embeds

# Replace with your endpoint if different
VLLM_SERVER_ENDPOINT = "http://localhost:8000/pooling"

def main():
    image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"

    request_payload_url = {
        "data": {
            "data": image_url,
            "data_format": "url",
            "image_format": "tiff",
            "out_data_format": "b64_json",
        },
        "priority": 0,
        "model": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    }

    ret = requests.post(VLLM_SERVER_ENDPOINT, json=request_payload_url)

    if ret.status_code != requests.status_codes:
        print(f"response.status_code: {ret.status_code}")
        print(f"response.reason:{ret.reason}")
        return

    response = ret.json()

    decoded_image = base64.b64decode(response["data"]["data"])
    out_path = Path.cwd()/"prithvi_prediction.tiff"

    with open(out_path, "wb") as f:
        f.write(decoded_image)

    print(f"Output file: {out_path!s}")

if __name__ == "__main__":
    main()
