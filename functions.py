from transformers import BlipProcessor , BlipForConditionalGeneration
from PIL import Image

def get_image_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    model_name = "Salesforce/blip-image-captioning-large"
    device = 'cpu'

    processor = BlipProcessor.from_pretrained(model_name)
    #processor = processor.to(device) 
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    inputs = processor(image , return_tensors='pt')
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0] , skip_special_tokens=True)
    
    return caption

def detect_objects(image_path):
    pass

if __name__ =='__main__':
    image_path = "C:/Users/jebalia/Desktop/OIP.jpg"
    caption = get_image_caption(image_path)
    print(caption)

