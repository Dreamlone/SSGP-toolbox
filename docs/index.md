<img src="https://raw.githubusercontent.com/Dreamlone/SSGP-toolbox/redesign/docs/media/images/label.png" width="800"/>

Welcome to [SSGP-toolbox](https://github.com/Dreamlone/SSGP-toolbox) documentation!

**Simple Spatial Gapfilling Processor toolbox** is an open-source library for processing 
remote sensing data. This is a simple and universal module for satellites images preprocessing 
and even performing some simple analysis (in addition to filling in the gaps).

## Key features 

- Ability to preprocess various remote sensing products such as `MOD11A1`, 
  `MOD11A2`, `MOD11_L2`, `MYD11A1`, `MYD11A2`, `MYD11_L2`, `SLSTR Level-2 LST` etc.:

    - Extractors (connectors to endpoints) which allow loading remote sensing products from well-known sources;
    - Perform preprocessing of spatial data and extract desired parameters from source archives;

- An algorithm based on cellular automata for detecting shaded (corrupted) pixels;
- Effective machine learning algorithm to fill in the gaps;
- Wrappers for preparing the output in a form of `netCDF`, `geotiff`, `npy` and many other files formats

## Documentation

Here is a detailed description of how the algorithm works, as well 
as many use cases, tutorials, and visualizations of the results. 
The documentation is organized into three large sections: 

- **Tutorials, guides and "how to" recipes** - provide examples and tips for 
  remote sensing data processing. Section divided into two subsections according to two versions 
  of interfaces: old (before 2023) and new (after 2023) one;
- **Algorithms explanation** - detailed explanation how the algorithms works;
- **Development** - contribution guides, library architecture description, roadmaps and related ideas

Check the table of contents for navigation on the mentioned above sections.

## Inspiration

*(informal message to colleagues)*

This library is made by engineers and researchers in the field of remote sensing data 
processing for engineers in exactly the same domain. So that colleagues 
can spend less time on boring preprocessing and more time on research! 

We know how much difficulty the existence of gaps in satellite images 
can cause, and we know how to fix it (and you probably too since now). 
