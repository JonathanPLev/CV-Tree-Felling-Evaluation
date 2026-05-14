# CV-Tree-Felling-Evaluation
We provide a repository for the training and implementation of a CV model that determines safety/viability of cutting down specific trees in a given forest area.

## Step 1: Image Processing and Bounding Box Generation
We used OpenCV to detect colored marker dots placed on trees in the field. Red dots are healthy trees (negative examples) and green dots are trees ready to be cut down (positive examples). 

Each detected dot is then used to generate a bounding box around the tree it is indicating. All the bounding boxes and the labels are exported in a JSON file. 