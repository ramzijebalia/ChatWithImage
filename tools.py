from langchain.tools import BaseTool
from transformers import BlipProcessor , BlipForConditionalGeneration , DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from fcts import get_image_caption , detect_objects

class ImageCaptionTool(BaseTool):
    name = "Image Captioner"
    description = "Use this tool when given the path of an image thatyou whould like to be describe"\
                   "return : short description of the image"
    def _run(self , img_path):
        caption = get_image_caption(img_path)
    
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("Error")


class ObjectDetectiionToool(BaseTool):
    name = "Object Detector"
    description = "Use this tool when given the path of an image that you whould like to detect objects"\
                    "return ; list of all detected objects ( coordinates of bounding box )"

    def _run(self , img_path):
        detections = object(img_path)
        
        return detections
    
    def _arun(self, query: str):
        raise NotImplementedError("Error")