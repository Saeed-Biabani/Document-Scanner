from Structure.model.Detector import GetModel
import numpy as np
import torch


class Scanner:
    def __init__(self, model_path, cfg):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GetModel(cfg).to(self.device)
        self.__loadweights__()


    def __loadweights__(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()


    def __normalization__(self, x) -> torch.Tensor:
        return ((x.float() - x.mean()) / x.std()).unsqueeze(0).unsqueeze(0)
    

    def ScanView(self, x):
        assert len(x.shape) == 2, "Image must be in gray scale!"
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device = self.device, dtype = torch.float32)

        x = self.__normalization__(x)
        pred = self.model(x)
        pred = torch.nn.Sigmoid()(pred)

        mask = pred.cpu().detach().numpy()[0]
        mask = np.moveaxis(mask, 0, -1)

        return np.where(mask > (mask.mean() + abs(mask.std() / 2)), 255, 0).astype("uint8")