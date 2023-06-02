import argparse
from torchvision import transforms
import onnxruntime as ort
from typing import Dict, Union, Tuple
import numpy as np
from kserve import Model, ModelServer, InferInput, InferRequest
import io
import base64
from PIL import Image


class ImageRetrieval(Model):
    def __init__(self, name: str, model_path: str):
       super().__init__(name)
       self.name = name
       self.load(model_path)


    def load(self, model_path: str):
        self.model = ort.InferenceSession(model_path,
                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model_path = "models/image_retrieval/image_retrieval/model.onnx"
        model = ort.InferenceSession(model_path,
                                     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.ready = True

    def image_processing(self, image: Union[str, bytes, bytearray, list]):
        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)

        # If the image is sent as bytesarray
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image)).convert("RGB")
            image = np.array(self.transform(image))

        elif isinstance(image, list):
            # if the image is a list
            image = np.asarray(image)
            image = np.array(self.transform(image))

        return image


    def predict(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> Dict:

        if isinstance(payload, InferRequest):
            input_tensors = [self.image_processing(self.name, instance) for instance in payload.inputs[0].data]
        else:
            input_tensors = [self.image_processing(self.name, instance["image"]["b64"]) for instance in
                             payload["instances"]]
        input_tensors = np.asarray(input_tensors, dtype=np.float32)


        output = self.model.run(None, {'input': image.numpy()})
        import time
        st = time.time()
        output = model.run(None, {'image_feature': np.expand_dims(input_tensors[0], 0)})
        et = time.time()
        print(et-st)


        request_id = str(payload.id) if isinstance(payload.id, str) else "N.A."
        infer_request = InferRequest(model_name=self.name, infer_inputs=[infer_inputs], request_id=request_id)


        output = self.model(input_tensor)
        torch.nn.functional.softmax(output, dim=1)
        values, top_5 = torch.topk(output, 5)
        result = values.flatten().tolist()
        response_id = generate_uuid()
        return {"predictions": result}

if __name__ == "__main__":
    model = AlexNetModel("custom-model")
    ModelServer().start([model])