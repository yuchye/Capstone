import {getPrediction} from "backend/serviceModule.jsw";

export function button1_click(event) {
	$w('#predictionText').text = "Computing predicted class - please be patient ...";
	getPrediction($w("#geneInput").value, $w("#variationInput").value, $w("#textInput").value)
	.then(result => $w('#predictionText').text = "Predicted Variation Class is " + result);
}
