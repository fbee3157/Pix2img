import requests
import nncf
from ov_janus_helper import convert_janus_model
from utils.notebook_utils import device_widget
from pathlib import Path
from modelscope import snapshot_download
from ov_janus_helper import OVJanusModel
from janus.models import VLChatProcessor
from gradio_helper import make_demo

# set CPU as the default hardware
device = device_widget("CPU", ["NPU"])

# select model
model_ids = ["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B", "deepseek-ai/Janus-1.3B"]

compression_configuration = {
    "mode": nncf.CompressWeightsMode.INT4_ASYM,
    "group_size": 64,
    "ratio": 1.0,
}


model_id = model_ids[0]

# download the model from modelscope
ms_model_id = model_id
ms_local_path  = "./deepseek-ai/Janus-Pro-1B"

if not Path(ms_local_path).exists():
    model_dir = snapshot_download(ms_model_id, cache_dir="./")
    
# convert the model
model_path = Path(model_id.split("/")[-1] + "-ov")
convert_janus_model(model_id, model_path, compression_configuration)

# load model and processor
ov_model = OVJanusModel(model_path, device.value)
processor = VLChatProcessor.from_pretrained(model_path)

# start Gradio APP
demo = make_demo(ov_model, processor)

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/