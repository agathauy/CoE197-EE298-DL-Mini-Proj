# CoE197-EE298-DL-Mini-Proj
Submitted by Francis Marquez and Agatha Uy as a mini project for CoE 197/EE 298-Z Deep Learning.


## What is this?
In this mini project, we explore the use case of generating music from a single or a few similar MIDI files. We want to aid the ideation process, as well as creation process, for people who have the need to generate MIDI files either for their video games, applications, etc. And for the case where they already own MIDI files that they want to reproduce for another use case. 

Our approach for now doesn't aim to generate an entire composition, but for someone to make use of the sequences generated by the network, for use with a DAWs like Garageband, FL Studio, Logic, and etc.

For this case, we explore the use of GANs and LSTM. However, due to us being beginners, we have decided to go for trying out simple networks first and build up on those. We went for this to fasten our learning with simple models first, along with how they tend to train much faster than stacked networks thus leading to faster feedbacks for us. We thought we could work on those and build up, however even with simple models given the time constraints we reached problems that we had to debug. Thus we settled on these models so far.

## What did we find out?
It is in fact possible to use just simple models to generate MIDI files that can be said to be similar to just one or a few sound samples. In our testings, bi-directional RNNs worked best with learning the melodies of the songs. The implementation of GANs on the otherhand did not quite capture the melodies of the songs. The generation of different samples for one song to another just takes around 5 to 15 minutes depending on length, while producing a generated song every 5 epochs.

We also observed that the LSTMs were having problems upon having too many states where it gets stuck in just having the same output over and over again. Thus, we note for future work that either another model, or an implementation with higher capacity is needed for an acceptable training time.

## Disclaimers
We should not that we do NOT own any of the original input midi files inside this repo, nor the repos inside music_generation_examples. Those were used as references, as well as were used in attempts to be the basis of our work.



# For Results
Below is a link to some of our results. The other generated files are in this repo.

https://soundcloud.com/agatha-uy/sets/deep-learning-mini-project-music-generation

