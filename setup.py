from setuptools import setup, find_packages
 
setup(
    name                = 'skprovider',
    version             = '0.1',
    description         = 'provide skeleton of a person',
    author              = 'LiiNen',
    author_email        = 'kjeonghoon065@gmail.com',
    url                 = 'https://github.com/LiiNen/CV_skeleton_provider',
    install_requires    =  ['numpy', 'opencv-python'],
    packages            = find_packages(exclude = []),
    keywords            = ['Skeleton Provider'],
    python_requires     = '>=3',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
