# Detect-Locate-CNN
A CNN that is able to detect and localise deformation in Sentinel-1 interferograms.  It uses data from [VolcNet](https://github.com/matthew-gaddes/VolcNet), and is described in detail in the EarthArXiv preprint (link pending).  Due to the large datasets required for deep learning, this repo contains a minimum working example that has much lower performance than that described in the publication.  

In short, it uses [SyInterferoPy](https://github.com/matthew-gaddes/SyInterferoPy) to synthesise unwrapped interferograms:
![01_Sample_of_synthetic_data](https://user-images.githubusercontent.com/10498635/104200533-b35f8380-5420-11eb-874a-1504d846eb52.png)

And loads ~250 real data from VolcNet:
![02_Sample_of_Real_data](https://user-images.githubusercontent.com/10498635/104200535-b3f81a00-5420-11eb-9e1c-61a4ce37b6d3.png)

This real data is then augmented to create ~650 real data (to be an equal match to the synthetic data):
![03_Sample_of_augmented_real_data](https://user-images.githubusercontent.com/10498635/104200538-b490b080-5420-11eb-9e4e-fb61bd64c527.png)

A model is trained to both determine the type of deformation, and the location of the deformation.  Note that accuracy cannot be applied to the localisation problem (i.e. there is no exact answer), so is omited from this figure.  
![04_ bottleneck_training](https://user-images.githubusercontent.com/10498635/104200540-b5294700-5420-11eb-9cfc-f7d7bb2ddc91.png)

The trained model can then be used to predict labels on the testing data.  Note that in this limited example, the performance is far lower than that achieved by the model described in the publication.  
![Testing_data_(after_step_06)](https://user-images.githubusercontent.com/10498635/104200541-b5c1dd80-5420-11eb-91de-09f9bb313c1f.png)

