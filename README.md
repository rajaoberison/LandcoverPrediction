# Prediction of Land Cover Change

This is still a work in progress. I stated working on this as part of the Geospatial Software Design class at Yale FES.

## Introduction
In this project, I'm trying to predict landcover change using simple random walk within landcover pixels. The software used was Earth Engine, and one of the main challenge was to incorporate pixel location information in the script. (And I'm still working on that part actually). This script example is specifically designed for mangrove cover change.

Mangroves are trees and shrubs that inhabit the interface between land and sea of the tropics and subtropics. Their natural distribution is limited, globally, by temperature (20°C winter isotherm of seawater), and, regionally and locally, by rainfall, tidal inundation, and freshwater inflow bringing nutrients and silt ([Kathiresan and Bingham, 2001](https://www.sciencedirect.com/science/article/pii/S0065288101400034); [Alongi and Brinkman, 2011](https://link.springer.com/chapter/10.1007/978-94-007-1363-5_10)). Additionally, mangroves are abundant in zones of small topographical gradients, well-drained soils, and large tidal amplitudes; but they do poorly in stagnant water ([Gopal and Krishnamurthy, 1993](https://link.springer.com/chapter/10.1007/978-94-015-8212-4_10); [Van Loon et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150302)).

<img align="right" width="474" height="294" src="https://github.com/rajaoberison/LandcoverPrediction/blob/master/images/mangrove_change.png">

Based on the assumption that only mangroves can tolerate intertidal areas, my method will assume that there are 6 types of landcover that each pixel can convert into, namely: mangroves, degraded mangroves, terrestrial forest, farming, sand/bare soil/urban, and water. Each state will convert to another state based on the probability transition matrix that will be calculated based on landcover classification, frequency of storm, upstream deforestation rate, proximity to human population and restoration project.

<br>

-------------
## Land Cover Classification
Getting accurate landcover class for the study area is crucial for this analysis, so I developped a code for landcover classification, which uses Landsat 5, elevation subset, Otsu segmentation, and random forest to produce binary class at each step.

<img align="left" width="501" height="262" src="https://github.com/rajaoberison/LandcoverPrediction/blob/master/images/flow_chart.png">

This land cover classification allowd me to produce a transition matrix of with probabilities of the conversion of each pixels from one state to another. This information is not enaough however, for the prediction analysis, because factors such as storm frequency, anthropogenic pressures, and upstream forest cover are not yet take into account. I will try to calculate this proability using Baysian inference.

<br>
<br>

---------------------

    But first let's simulate a simple random walk using the classes and the transition matrix obtained from the 
    classification. By choosing a study region in Belo-sur-Tsiribihina, Madagascar, and a timeframe of 2000 to 
    2010 (with a two-year intervall), the classification determined that the current state is: 
    
    In short, the script looks like this:
    
```javascript
// DEFINING FUNCTIONS
// OTSU FUNCTION
// https://medium.com/google-earth/otsus-method-for-image-segmentation-f5c48f405e
var otsu = function(histogram) {…};


// RANDOM FOREST CLASSIFIER GIVEN TRAINING REGIONS
var RFclassifier = function(image, training0, training1, trainingbands){…};


// LANDSAT 5 IMAGE CLASSIFIER
var l5classifier = function(year, aoi, training_region){…};


// LAND COVER CLASSIFICATION GIVEN THE REGION OF STUDY
// From 2000 to 2008 with a two-year intervall
var cover_2000 = l5classifier(2000, belo, training);
var cover_2002 = l5classifier(2002, belo, training);
var cover_2004 = l5classifier(2004, belo, training);
var cover_2006 = l5classifier(2006, belo, training);
var cover_2008 = l5classifier(2008, belo, training);
```

Typical outputs


<img align="left" width="332" height="480" src="https://github.com/rajaoberison/LandcoverPrediction/blob/master/images/alltheway.gif">


## Calculation of Posterior Probabilities


