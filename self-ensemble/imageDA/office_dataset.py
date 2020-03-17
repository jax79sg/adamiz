import os
import urllib
from imageDA import settings
from imageDA.image_dataset import ImageDataset
import requests
from tqdm import tqdm
import tarfile


class OfficeDataset (ImageDataset):
    def __init__(self, img_size, range01, rgb_order, images_dir, dummy=False, download=True):
        class_names = []
        names = []
        paths = []
        y = []

        for dir_name in sorted(list(os.listdir(images_dir))):
            cls_dir_path = os.path.join(images_dir, dir_name)
            if os.path.isdir(cls_dir_path):
                cls_i = len(class_names)
                class_names.append(dir_name)

                for file_name in os.listdir(cls_dir_path):
                    file_path = os.path.join(cls_dir_path, file_name)
                    if os.path.isfile(file_path):
                        name = dir_name + '/' + file_name
                        names.append(name)
                        paths.append(os.path.join(cls_dir_path, file_name))
                        y.append(cls_i)

        super(OfficeDataset, self).__init__(img_size, range01, rgb_order, class_names, len(class_names),
                                             names, paths, y, dummy=dummy)


class OfficeAmazonDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False, download=True):

        self.file_id = '0B4IapRTv9pJ1WGZVd1VDMmhwdlE'
        self.filename = "domain_adaptation_images.tar.gz"
        # download dataset.
        if download:
            download_file_from_google_drive(self.file_id, os.path.join('./dataset/', self.filename))
        if not _check_exists(os.path.join('./dataset/', self.filename)):
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        load_samples(os.path.join('./dataset/', self.filename))

        images_dir = settings.get_data_dir('office_amazon')
        super(OfficeAmazonDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)


class OfficeDSLRDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        images_dir = settings.get_data_dir('office_dslr')
        super(OfficeDSLRDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)


class OfficeWebcamDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        images_dir = settings.get_data_dir('office_webcam')
        super(OfficeWebcamDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)
        
class OfficeAmazon2DslrDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        images_dir = settings.get_data_dir('amazon_dslr')
        super(OfficeAmazon2DslrDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)

class OfficeAmazon2WebcamDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        images_dir = settings.get_data_dir('amazon_webcam')
        super(OfficeAmazon2WebcamDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)

class OfficeDslr2AmazonDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        images_dir = settings.get_data_dir('dslr_amazon')
        super(OfficeDslr2AmazonDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)

class OfficeDslr2WebcamDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        images_dir = settings.get_data_dir('dslr_webcam')
        super(OfficeDslr2WebcamDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)

class OfficeWebcam2AmazonDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        images_dir = settings.get_data_dir('webcam_amazon')
        super(OfficeWebcam2AmazonDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)

class OfficeWebcam2DslrDataset (OfficeDataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        images_dir = settings.get_data_dir('webcam_dslr')
        super(OfficeWebcam2DslrDataset, self).__init__(img_size, range01, rgb_order, images_dir, dummy=dummy)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
        total_length = response.headers.get('content-length')

    save_response_content(response, destination,total_length)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def load_samples(filename):
    """Load sample images from dataset."""
    filename = filename
    print('unzipping the file')
    with tarfile.open(filename, 'r') as zip_ref:
        zip_ref.extractall('./dataset/office31')
    '''
    f = gzip.open(filename, "rb")
    data_set = pickle.load(f, encoding="bytes")
    f.close()
    if self.train:
        images = data_set[0][0]
        labels = data_set[0][1]
        self.dataset_size = labels.shape[0]
    else:
        images = data_set[1][0]
        labels = data_set[1][1]
        self.dataset_size = labels.shape[0]
    return images, labels
    '''


def save_response_content(response, destination,total_length):
    CHUNK_SIZE = 32768
    datasize = len(response.content)
    t=tqdm(total=datasize)


    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):

            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                t.update(len(chunk))
    t.close()

def _check_exists(filename):
    """Check if dataset is download and in right place."""
    #return os.path.exists(os.path.join(self.root, self.filename))
    return os.path.exists(os.path.join('.', filename))
