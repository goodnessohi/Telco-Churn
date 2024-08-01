from setuptools import setup, find_packages

setup(
    name='Telco-Churn',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        
         'catboost==1.1.1',
         'scikit-learn==1.1.2',
         'pandas==1.4.3',
          'numpy'  
            ],
    author='Ohimai',
    author_email='goodnessobaika@gmail.com'
)