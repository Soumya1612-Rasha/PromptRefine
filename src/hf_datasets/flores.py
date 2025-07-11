import json
import datasets

_HOMEPAGE = " "
_LICENSE= " "
_CITATION=" "
_DESCRIPTION=" "
import pdb

all_resource_language={
    'Awadhi':'awa',
    'Haryanvi': 'bgc',
    'Tibetan':'bo',
    'Garhwali':'gbm',
#     'Konkani':'kon',
    'Chhattisgarhi':'hne',
    'Rajasthani':'hoj',
    'Maithili':'mai',
    'Manipuri':'mni',
    'Malvi':'mup',
    'Marwari':'mwr',
    'Santali':'sat',
    'Bodo':'brx',
    'Bengali':'bn',
    'Gujarati':'gu',
    'Hindi':'hi',
    'Kannada':'kn',
    'Malayalam':'ml',
    'Marathi':'mr',
    'Tamil':'ta',
    'Telugu':'te',
    'Urdu':'ur',
    'English':'en'}

class FLORESConfig(datasets.BuilderConfig):
    def __init__(self, lang_name=None, high_lang_name=None, all_data = None, test_lang_name=None, **kwargs):
        super().__init__(**kwargs)
        
        self.lang_name = lang_name
        self.high_lang_name = high_lang_name
        self.test_lang_name = test_lang_name
        self.all_data = all_data

class FLORES(datasets.GeneratorBasedBuilder):
    """The FLORES dataset"""
    BUILDER_CONFIGS = [
        FLORESConfig(
            name="flores",
            version=datasets.Version("1.0.0"),
            description="FLORES dataset with custom language codes",

        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "source": datasets.Value("string"),
                    "target": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        
        """Return SplitGenerators"""
        downloaded_files="."
        # pdb.set_trace()
        lang_name = self.config.lang_name
        high_lang_name = self.config.high_lang_name
        test_lang_name = self.config.test_lang_name

        all_data = self.config.all_data
        
        lang_code = all_resource_language[lang_name]
       
        test_lang_code = all_resource_language[test_lang_name]
        print("**********************")
        print("Using the following settings: \n\n")
        print(f"Language = {lang_name}\n Aux Language = {high_lang_name}\n Test Language={test_lang_name}\n Use All Data={all_data}")
        if all_data:
            print("USING AUXILIARY DATA!!!! \n STOP IF NOT INTENDED \n *********************")
            high_lang_code = [all_resource_language[name] for name in high_lang_name]
            high_lang_name = high_lang_name
            

            print(f"LOADING FOLLOWING LANGUAGES:\n 1. {lang_code} \n 2. {high_lang_code}")
            final_path_name_prefix = f"flores_{lang_code}_en_"
            for code in high_lang_code:
                    final_path_name_prefix = final_path_name_prefix + f"{code}_en_"
            
            final_path_name_prefix = final_path_name_prefix + "train.json"
            
            print(final_path_name_prefix)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": downloaded_files + f"/flores_in/{final_path_name_prefix}"}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": downloaded_files + f"/flores_in/flores_{test_lang_code}_en_dev.json"}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": downloaded_files + f"/flores_in/flores_{test_lang_code}_en_test.json"}
                )
            ]
    
        else:
            print(f"LOADING FOLLOWING LANGUAGES:\n 1. {lang_code}")


            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": downloaded_files + f"/flores_in/flores_{lang_code}_en_train.json"}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": downloaded_files + f"/flores_in/flores_{test_lang_code}_en_dev.json"}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": downloaded_files + f"/flores_in/flores_{test_lang_code}_en_test.json"}
                )
            ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath) as fin:
            data_ = json.load(fin)
            data = data_["examples"]
            for i, line in enumerate(data):
                entry = {
                    "source": line["source"],
                    "target": line["target"],
                }
                yield i, entry