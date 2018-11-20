package br.clustering.util;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Util {

	public static final Map<String, Integer> targets = createMap();
    private static Map<String, Integer> createMap()
    {
        Map<String,Integer> myMap = new HashMap<>();
        myMap.put("grass", 0);
        myMap.put("path", 1);
        myMap.put("window", 2);
        myMap.put("cement", 3);
        myMap.put("foliage", 4);
        myMap.put("sky", 5);
        myMap.put("brickface", 6);
        return myMap;
    }

    /**
     * Normalize the values in each column in the range between 0 and 1.
     * @param values - List<List<Double>>
     * @return retMat - Double[][]
     */
	public static Double[][] normalize(List<List<Double>> values){
		final int HEIGHT = values.size();
		final int WIDTH = values.get(0).size();
		
		Double[][] retMat = new Double[HEIGHT][WIDTH];
		Double[] max = new Double[WIDTH]; 
		Double[] min = new Double[WIDTH];
		
		for (int i = 0; i < min.length; i++) {
			min[i] = Double.POSITIVE_INFINITY;
			max[i] = Double.NEGATIVE_INFINITY;
		}
		for (List<Double> rows : values) {
			for (int i = 0; i < rows.size(); i++) {
				Double feature = rows.get(i);
				if(max[i] < feature) max[i] = feature;
				if(min[i] > feature) min[i] = feature;
			}
		}
		
		for (int i = 0; i < retMat.length; i++) {
			for (int j = 0; j < retMat[0].length; j++) {
				retMat[i][j] = (values.get(i).get(j) - min[j]) / (max[j] - min[j]);
			}
		}
		
		return retMat;
	}
}
