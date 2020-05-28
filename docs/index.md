# Introduction
In this project, I'm trying to predict landcover change using simple random walk within landcover pixels. The software used was Earth Engine, and one of the main challenge was to incorporate pixel location information in the script. (And I'm still working on that part actually). This script example is specifically designed for mangrove cover change.

Mangroves are trees and shrubs that inhabit the interface between land and sea of the tropics and subtropics. Their natural distribution is limited, globally, by temperature (20°C winter isotherm of seawater), and, regionally and locally, by rainfall, tidal inundation, and freshwater inflow bringing nutrients and silt ([Kathiresan and Bingham, 2001](https://www.sciencedirect.com/science/article/pii/S0065288101400034); [Alongi and Brinkman, 2011](https://link.springer.com/chapter/10.1007/978-94-007-1363-5_10)). Additionally, mangroves are abundant in zones of small topographical gradients, well-drained soils, and large tidal amplitudes; but they do poorly in stagnant water ([Gopal and Krishnamurthy, 1993](https://link.springer.com/chapter/10.1007/978-94-015-8212-4_10); [Van Loon et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150302)).

<img align="right" width="474" height="294" src="https://rajaoberison.github.io/images/landcoverpred/mangrove_change.png">

Based on the assumption that only mangroves can tolerate intertidal areas, my method will assume that there are 6 types of landcover that each pixel can convert into, namely: mangroves, degraded mangroves, terrestrial forest, farming, sand/bare soil/urban, and water. Each state will convert to another state based on the probability transition matrix that will be calculated based on landcover classification, frequency of storm, upstream deforestation rate, proximity to human population and restoration project.

<br/>

-------------
# Land Cover Classification
Getting accurate landcover class for the study area is crucial for this analysis, so I developped a code for landcover classification, which uses Landsat 5, elevation subset, Otsu segmentation, and random forest to produce binary class at each step.

<img align="left" width="501" height="262" src="https://rajaoberison.github.io/images/landcoverpred/flow_chart.png">

This land cover classification allowd me to produce a transition matrix of with probabilities of the conversion of each pixels from one state to another. This information is not enaough however, for the prediction analysis, because factors such as storm frequency, anthropogenic pressures, and upstream forest cover are not yet take into account. I will try to calculate this proability using Baysian inference.

<br/><br/>

---------------------
# Examples simulation

But first let's simulate a simple random walk using the classes and the transition matrix obtained from the 
classification. By choosing a study region in Belo-sur-Tsiribihina, Madagascar, and a timeframe of 2000 to 
2010 (with a two-year intervall), I obtained the following outputs. For this example, water was not yet 
included but just the land covers.

In short, the script looks like this:
    
```javascript
// Attempted Land Cover Change Prediction by Andry Rajaoberison
// NB: This code is to provide insights on how to do prediction analysis or randow-walk
// using Google Earth Engine. No accuracy assessment were conducted and the training
// for classification were based on obsevation of high resolution Google Earth Imagery.


// SCALE OF THE STUDY
var scale = 30; // meters



/** DEFINING FUNCTIONS **/
/*** OTSU FUNCTION ***/
// https://medium.com/google-earth/otsus-method-for-image-segmentation-f5c48f405e
var otsu = function(histogram) {<==>};

 
/*** RANDOM FOREST CLASSIFIER GIVEN TRAINING REGIONS ***/
var RFclassifier = function(image, training0, training1, trainingbands, scale){<==>};


/*** LANDSAT 5 IMAGE CLASSIFIER ***/
var l5classifier = function(year, aoi, training_region, scale){<==>};


/*** TRANSITION MATRIX CALCULATOR ***/
var transition_matrix = function(before_image, current_image, year, aoi, scale){<==>};


/*** RANDOM WALK FUNCTION ***/
// This requires a transition matrix which is calculated above.
// For each of the pixels, the current state is given by the rows of the average matrix
// Then, the next state of the land cover is given by the result of product of 
// current state * average transition matrix (within the timeframe)
// As the current state is a 1D array (vector), the product will occur for each column
// of the average matrix, whcih means, we have to get it's transposed version
// Here's the function for all of that
var random_walk = function(current_cover, bandNameOfClasses, average_matrix){<==>};



/** MAIN CODE **/
/*** LAND COVER CLASSIFICATION GIVEN THE REGION OF STUDY ***/<==>

/*** COMPUTING TRANSITION MATRIX ***/<==>


/*** RANDOM WALKING ***/ <==>
// First we need the average of the transition matrices
/**** AVERAGE TRANSITION MATRIX ****/<==>

/**** SIMULATION FROM 2008 to 2010 ****/<==>



/** PRESENTATION **/<==>
```

Here are some typical outputs:

--------------------

<img align="left" width="31%" src="https://rajaoberison.github.io/images/landcoverpred/actual.gif">
<img align="left" width="31%" src="https://rajaoberison.github.io/images/landcoverpred/fromprevious.gif">
<img align="left" width="31%" src="https://rajaoberison.github.io/images/landcoverpred/alltheway.gif">

As we can see, what the script does is: it will assign for all landcovers of type "a" to some new  
land cover of type "b". So it will convert everything, all mangroves to some land cover, all 
terrestrial forests to some land cover, and so on. While visually, it produces results a little off
from the actual land cover, the scrpits still provide insights into when where the mangroves
vulnerables within the timeframe of study.

If you look closely at the far-right simulation, mangroves are completely lost at the early 2000 
but then come back around 2010, I think this is because of the rate of mangrove loss higher 
in the early 2000 and slower around 2010. Obviously, a way to correct this sript is to 
incorporate some spatial information in the calculation of the probabilities, such as proximity  
of the land cover to population centers, proximity to coastline, frequency of storms, and 
upstream land cover (all of which may affect mangrove change).
The next step of this script will try to incorporate these information.
    

---------------------
# Calculation of Posterior Probabilities
For the next steps, the updating of the landcover class (the random walk) will go by pixels and not by land cover type.
This is something, I still find challenging to implement on Earth Engine as I haven't mastered some of its capabilities yet.

Please contact me if you're interested in collaborating on this project.
