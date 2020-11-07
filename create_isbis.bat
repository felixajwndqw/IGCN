python .\isbi.py --produce --path models/seg/isbi/isbi_kernel_size=3_no_g=5_base_channels=10_epoch216.pk
python .\isbi.py --produce --path models/seg/isbi/isbi_kernel_size=5_no_g=5_base_channels=10_epoch113.pk
python .\isbi.py --produce --path models/seg/isbi/isbi_kernel_size=7_no_g=5_base_channels=10_epoch244.pk
python .\isbi.py --produce --path models/seg/isbi/isbi_kernel_size=9_no_g=5_base_channels=10_epoch208.pk
python .\data.py --post --dir data\isbi\test\labels\isbi_kernel_size=3_no_g=5_base_channels=10_epoch216.pk
python .\data.py --post --dir data\isbi\test\labels\isbi_kernel_size=5_no_g=5_base_channels=10_epoch113.pk
python .\data.py --post --dir data\isbi\test\labels\isbi_kernel_size=7_no_g=5_base_channels=10_epoch244.pk
python .\data.py --post --dir data\isbi\test\labels\isbi_kernel_size=9_no_g=5_base_channels=10_epoch208.pk