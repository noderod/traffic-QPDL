# Traffic Patterns Prediction



# Summary

This project attempts to predict the AADT traffic volumes for each road within a road network, first utilizing a quadratic programming (QP)
optimizer, secondly by using neural networks.

This project was developed as the project for CS 6140 (Machine Learning), completed during the Spring 2021 semester at Northeastern
University.


# Approach

Unless specified, all commands will be assumed to run from this project's main directory.


## Data origin

This project utilizes map data obtained from OpenStreetMap, which is exported using the same XML format provided by OpenStreetMap.
AADT values must be provided as a list using the following JSON format (ID tag may be excluded):
```json
{
    "AADT":[
        {
            "ID":"a1",
            "latlon":[0, 0],
            "AADT μ":100,
            "AADT σ":20
        },
        {
            "ID":"b2",
            "latlon":[0, 1],
            "AADT μ":200,
            "AADT σ":40
        }
    ]
}
```

Where:
- μ: AADT mean
- σ: AADT standard deviation


## Quadratic Programming optimization

### Data pre-processing

The OpenStreetMap and AADT data must be preprocessed into a JSON output via:
```bash
python3 QP/preprocessing.py \
    --osm $OSM_FILE_FILEPATH \
    --aadt $AADT_JSON_FILEPATH \
    --output $OUTPUT_JSON_FILEPATH \
```

For example:
```bash
python3 QP/preprocessing.py \
    --osm examples/Niles/Niles.osm \
    --aadt examples/Niles/Niles_AADT.json  \
    --output test/example1.json
```

Use the *--verbose* flag to show a map as well as node and road counts.


If requesting information about how the flags in more detail, run:
```bash
python3 QP/preprocessing.py --help
```

### Convex optimization:

Run the quadratic optimizer (OSQP solver) using the above JSON preprocessed output (note, the *--verbose* flag will have no effect on the
actual output) in order to obtain a JSON output which will contain nodes and road information, as well as the AADT calculated for each road.
Run via:
```bash
python3 QP/QP.py \
    --input $PREPROCESSING_OUTPUT_FILEPATH \
    --output $OUTPUT_JSON_FILEPATH
```


Assuming an output being *test/example1.json*, run via:

```bash
python3 QP/QP.py \
    --input test/example1.json \
    --output test/output1.json
```


Use the *--verbose* flag to show a map of each road, colorcoded depending on the AADT value.


If requesting information about how the flags in more detail, run:
```bash
python3 QP/QP.py --help
```

The solver has been found to return inaccurate results when enforcing strict constraints when setting *adaptive_rho = False*[4]. Therefore, this
has not been strictly enforced.


## Neural networks






# Licensing

This project utilizes the OSPQ solver, which uses the [Apache License 2.0](https://github.com/oxfordcontrol/osqp/blob/master/LICENSE), a copy of this license is also provided in this repository [here](./licensing/Apache_license_2.txt).


## Data examples

The provided examples were obtained using data from:
* Niles
	* AADT values obtained from [Michigan DOT 2013 ADT](https://mdotcf.state.mi.us/public/maps_adtmaparchive/listfiles.cfm?folder=2013adt)
	* Coordinate box: {"N": 41.8751, "W":-86.3211, "S":41.7903, "E":-86.1709}
	* AADT coordinates obtained from OpenStreetMap






Map data copyrighted OpenStreetMap contributors and available from [https://www.openstreetmap.org](https://www.openstreetmap.org).



# References

1. [https://stackabuse.com/reading-and-writing-xml-files-in-python/](https://stackabuse.com/reading-and-writing-xml-files-in-python/)
2. [https://stackoverflow.com/questions/17390166/python-xml-minidom-get-element-by-tag-in-child-node](https://stackoverflow.com/questions/17390166/python-xml-minidom-get-element-by-tag-in-child-node), question
3. [https://mdotcf.state.mi.us/public/maps_adtmaparchive/listfiles.cfm?folder=2013adt](https://mdotcf.state.mi.us/public/maps_adtmaparchive/listfiles.cfm?folder=2013adt)
4. [https://github.com/oxfordcontrol/OSQP.jl/issues/47](https://github.com/oxfordcontrol/OSQP.jl/issues/47)
5. [https://github.com/noderod/City-Learning/blob/master/NY_fire_inspection/NY_Visualizer.py](https://github.com/noderod/City-Learning/blob/master/NY_fire_inspection/NY_Visualizer.py)
