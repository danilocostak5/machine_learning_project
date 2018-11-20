package br.clustering.util;

import java.util.HashMap;
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
}
