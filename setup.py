from setuptools import setup, find_packages
setup(name = "DeepNLP" ,
      version = "0.1" ,
      description = "DeepNLP is a library for Natural Language Processing in python." ,
      author = "CUHKSZ-NLP Group" ,
      url = "https://github.com/yuanheTian/dnlptk" ,
      license = " LGPL " ,
      packages = find_packages(include=["DeepNLP", "DeepNLP.*"]),
      py_modules=["DeepNLP"])
