import torch
from src.uap_results import UAPSingleRes

class UapBasicLP:
    def __init__(self, net, props, args, baseline_results) -> None:
        self.net = net
        self.props = props
        self.args = args
        self.baseline_results = baseline_results


    def run(self) -> UAPSingleRes:
        return None