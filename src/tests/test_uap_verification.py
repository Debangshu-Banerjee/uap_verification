
from unittest import TestCase
from src import config
from src.specs.property import InputSpecType
from src.common import Domain
from src.common.dataset import Dataset
import src.uap_analyzer as uap_ver


class TestBasicUap(TestCase):
    def test_mnist_uap(self):
        uap_verfication_args = uap_ver.UapAnalysisArgs(
            domain=Domain.DEEPZ_UAP, baseline_domain=Domain.DEEPZ, dataset=Dataset.MNIST, sink_label=0,
            spec_type=InputSpecType.UAP, count=10, count_per_prop=20, eps=0.1, net=config.MNIST_FFN_01,
            timeout=100, output_dir='results/', radius_l=0.05, radius_r=0.25)
        uap_ver.UapVerification(uap_verfication_args)