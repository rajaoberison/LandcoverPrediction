# Prediction of Land Cover Change

This is still a work in progress. I stated working on this as part of the Geospatial Software Design class at Yale FES.

## Introduction
In this project, I'm trying to predict landcover change using simple random walk within pixels. The software used was Earth Engine, and one of the main challenge was to incorporate pixel location information in the script. (And I'm still working on that part actually). This script example is specifically designed for mangrove cover change.

Mangroves are trees and shrubs that inhabit the interface between land and sea of the tropics and subtropics. Their natural distribution is limited, globally, by temperature (20°C winter isotherm of seawater), and, regionally and locally, by rainfall, tidal inundation, and freshwater inflow bringing nutrients and silt ([Kathiresan and Bingham, 2001](https://www.sciencedirect.com/science/article/pii/S0065288101400034); [Alongi and Brinkman, 2011](https://link.springer.com/chapter/10.1007/978-94-007-1363-5_10)). Additionally, mangroves are abundant in zones of small topographical gradients, well-drained soils, and large tidal amplitudes; but they do poorly in stagnant water ([Gopal and Krishnamurthy, 1993](https://link.springer.com/chapter/10.1007/978-94-015-8212-4_10); [Van Loon et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150302)).

<img align="right" width="474" height="294" src="https://github.com/rajaoberison/LandcoverPrediction/blob/master/images/mangrove_change.png">

Based on the assumption that only mangroves can tolerate intertidal areas, my method will assume that there are 6 types of landcover that each pixel can convert into, namely: mangroves, degraded mangroves, terrestrial forest, farming, sand/bare soil/urban, and water. Each state will convert to another state based on the probability transition matrix that will be calculated based on landcover classification, frequency of storm, upstream deforestation rate, proximity to human population and restoration project.

-------------

## Land Cover Classification
Geeting accurate landcover class for the study area is crucial for this analysis, so I developped a code for landcover classification, which uses Landsat 5, elevation subset, Otsu segmentation, and random forest to produce binary class at each step.

<img align="left" width="474" height="294" src="https://github.com/rajaoberison/LandcoverPrediction/blob/master/images/flow_chart.png">
