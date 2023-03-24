
from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.uap_analyzer as uap_ver


class TestBasicUap(TestCase):
    def test_mnist_uap(self):
        eps = 0.02
        for i in range(10):
            uap_verfication_args = uap_ver.UapAnalysisArgs(
                domain=Domain.DEEPZ_UAP, baseline_domain=Domain.DEEPZ, dataset=Dataset.MNIST, 
                spec_type=InputSpecType.LINF, count=50, count_per_prop=5, eps=eps, net=config.MNIST_FFN_L2,
                timeout=100, output_dir='results/')
            uap_ver.UapVerification(uap_verfication_args)
            eps += 0.005