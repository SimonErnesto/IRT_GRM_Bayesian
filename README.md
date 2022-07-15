<h1> CHS Analysis GRM </h1>

<p>
	The present analysis implements a Bayesian graded response model (GRM) to estimate the item response to 44 items in a cybersecurity habits scale (CHS), where the latent trait (&psi;) is understood to be participants' cybersecurity habits (i.e. how secure is their behaviour when using electronic devices such as phones, computers, etc.). Although the scale incorporates 3 dimensions (blocking, hiding, inspecting) representing possible types of cybersecurity behaviours/habits, the present model does not make any assumption about differences across these dimensions.  
</p>
<p></p>

<h1> Model </h1>

<p> The model follows a conventional GRM approach, with a discrimination parameter ranging over items (&delta;<sub>i</sub>), a latent trait parameter ranging over subjects/participants (&psi;<sub>s</sub>), and a conventionality parameter ranging over items and ordered categories of scores (&kappa;<sub>i,c</sub>), which indicates how common/uncommon it is to engage in the behaviour described by the item (this is referred to as difficulty when a test involves correct/incorrect responses). In other words, an item answered with the highest score by most participants, irrespective of their cybersecurity habits level, would indicate behaviour that is too conventional and vice-versa. In the present case the discrimination parameter (&delta;) is only multiplied by the participants' trait parameter (&psi;); which provided better convergence and fit. Reasonable if we think of &delta; and &kappa; as the scale and location of &psi; respectively. The model is completed by an ordered logistic distribution over the estimated parameter (y&#770;). </p>

<p align="center"> &delta;<sub>i</sub> ~ Log-normal(0, 0.5), item<sub>i=1</sub>...item<sub>i=44</sub> </p>
<p align="center"> &psi;<sub>s</sub> ~ Normal(0, 0.5), subject<sub>s=1</sub>...subject<sub>s=134</sub> </p>
<p align="center"> &mu; ~ Normal(0, 0.05) </p>
<p align="center"> &sigma; ~ Half-normal(0.5) </p>
<p align="center"> &kappa;<sub>i,</sub><sub>c</sub> ~ Normal(&mu;, &sigma;), item<sub>i=1</sub>...item<sub>i=44</sub>, cutpoint<sub>c=1</sub>...cutpoint<sub>i=C-1</sub>, C=5 </p> 
<p align="center"> &eta; = &delta;<sub>i</sub>&psi;<sub>s</sub> </p>
<p align="center"> logit<sup>-1</sup>(x) = 1/(1 + e<sup>-x</sup>) </p>
<p align="center"> y&#770; = 1 - logit<sup>-1</sup>(&eta; - &kappa;<sub>1</sub>), if c = 0 </p>
<p align="center"> y&#770; = logit<sup>-1</sup>(&eta; - &kappa;<sub>c-1</sub>) - logit<sup>-1</sup>(&eta; - &kappa;<sub>c</sub>), if 0 < c < C </p>
<p align="center"> y&#770; = logit<sup>-1</sup>(&eta; - &kappa;<sub>c-1</sub>), if c = C, C = 5 </p>

<p align="center">
	<img src="model.png" width="500" height="500" />
<p>

<p> Before running the model, we ensured that priors were weakly informative but sensible via a prior predictive check. With no specific expectations per item, we placed the greater probability on the median of the Likert scale (2 points). Image below gives an example of prior predictive check probability over item 33. </p>

<p align="center">
	<img src="prior_preds/inspecting_q33_prob.png" width="600" height="400" />
</p>

<h1> Results </h1>

<p> We sampled the model using Markov chain Monte Carlo (MCMC) No U-turn sampling (NUTS) with 2000 tuning steps, 2000 samples, 4 chains and 0.99 acceptance target. The model sampled well, with 1.01 > R&#770; > 0.99;  BFMIs > 0.8, and bulk ESS > 1800 for all parameters. Posterior predictive checks show excellent predictive capacity, as indicated in the image below.  </p>

<p align="center">
	<img src="ppc.png" width="600" height="400" />
</p>

<p> Expectations show very good precision (max ~15%), though slightly higher than previous structural equations models (https://github.com/SimonErnesto/bsem_precision_analysis). An example in the image below. </p>

<p align="center">
	<img src="response_prob/inspecting_q33_prob.png" width="600" height="400" />
</p>

<p> Informative items should present item characteristic curves (ICCs) where the probability of giving low scores to an item should be low if a participant has low cybersecurity habits (&psi;) and vice-versa. The ICC curve is simply the expected probability of the model ranging across the estimated cybersecurity habits parameter (&psi;). Image below shows an example of an informative item, where low cybersecurity habits are associated with a greater probability (68%) of giving a 0 score, while high cybersecurity habits are associated with a high probability (63%) of giving a 4 score. </p>

<p align="center">
	<img src="item_characteristics/inspecting_q33_char.png" width="600" height="400" />
</p>

<p> Similarly, informative items should present item information curves (ITCs) where the probability of giving low scores to an item should be low if a participant has low cybersecurity habits (&psi;) and vice-versa. The ITC curve follows the function I(&psi;) = &delta;<sup>2</sup>(p(1-p)), where p = logit<sup>-1</sup>(&delta;<sub>i</sub>&psi;<sub>s</sub> - &kappa;<sub>i,</sub><sub>c</sub>) , ranging across the estimated cybersecurity habits parameter (&psi;). Image below shows an example of an informative item, where information peaks per score concentrate at increasingly higher values of &psi;. This indicates that item's scores are informative respect to participants' cybersecurity levels. </p>

<p align="center">
	<img src="item_information/inspecting_q33_info.png" width="600" height="400" />
</p>

<p> Finally, the test characteristic curve (TCC: sum of expected probability across items) and the test information curve (TIC: &Sigma;<sup>S</sup><sub>s=1</sub> = I</sub><sub>s</sub>(&psi;) ) indicate that, overall, the CHS scale provides good information about cybersecurity habits. Images below show TCC and TIC. TCCs indicates a stronger preference for higher scores (3 and 4) when cybersecurity habits are high, but a high overlap of scores when cybersecurity habits are low. TICs suggest that scores are generally informative at sensible cybersecurity habits values (chv), with score 0 peaking (~20 info) between -1.5 and -1.0 chv, score 1 peaking (~25 info) between -1.0 and -0.5 chv, score 2 peaking (~20 info) between -0.5 and 0.0 chv, score 3 peaking (~28 info) around 0.5 chv, and score 4 peaking (~25 info) between 0.5 and 1.0 chv. Namely, low scores appropriately represent low habits and vice-versa. </p> 

<p float='left' align="center">
	<img src="item_characteristics/test_char.png" width="400" height="300" />
	<img src="item_information/test_info.png" width="400" height="300" />
</p>

<h1> Conclusion </h1>  

<p> Present results indicate that the present cybersecurity habits scale (CHS) is generally effective for inferring and predicting cybersecurity habits. Even so, most items seem to be lowly informative, which may suggest that reducing the scale is a relevant further step. CHS can be tuned to become a useful instrument for measuring cybersecurity behaviours regarding cyber-hygiene habits. </p>