# predict.py
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
import requests
from io import BytesIO
from options.base_options import BaseOptions
from models import create_model
from data_loader import get_transform
from util import util

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Postavljanje opcija
        opt = BaseOptions().parse()
        opt.isTrain = False  # Set to False for inference
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()

    def predict(
        self,
        crop: bool = False,
        seed: int = 42,
        steps: int = 30,
        category: str = "upper_body",
        force_dc: bool = False,
        garm_img: str,
        human_img: str,
        mask_only: bool = False,
        garment_des: str = "cute pink top"
    ) -> Path:
        torch.manual_seed(seed)

        # Preuzimanje slika sa URL-ova
        garm_img = Image.open(BytesIO(requests.get(garm_img).content)).convert("RGB")
        human_img = Image.open(BytesIO(requests.get(human_img).content)).convert("RGB")
        
        # Pretvaranje slika u tensor
        transform = get_transform()
        garm_img = transform(garm_img).unsqueeze(0).to(self.device)
        human_img = transform(human_img).unsqueeze(0).to(self.device)

        # Priprema dodatnih parametara
        params = {
            'crop': crop,
            'steps': steps,
            'category': category,
            'force_dc': force_dc,
            'mask_only': mask_only,
            'garment_des': garment_des,
        }

        # Predikcija
        with torch.no_grad():
            input_data = {'label': human_img, 'image': garm_img, **params}
            self.model.set_input(input_data)
            self.model.test()
            visuals = self.model.get_current_visuals()
            output = visuals['fake_image']

        # Sačuvajte rezultat
        output_image = util.tensor2im(output)
        output_image = Image.fromarray(output_image)
        output_path = "output.png"
        output_image.save(output_path)

        return Path(output_path)
