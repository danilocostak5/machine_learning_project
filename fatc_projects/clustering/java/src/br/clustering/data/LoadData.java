package br.clustering.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import br.clustering.util.Util;

public final class LoadData {
	private static final LoadData INSTANCE = new LoadData();
	private static BufferedReader bfr = null;

	// Private method.
	private LoadData() {
	}

	// Returns my instance.
	public static LoadData getInstance() {
		return INSTANCE;
	}

	/**
	 * Receives a path to the segmentation csv-like file.
	 *
	 * @param path
	 * @return A map <id, Map<y, X>> with the the id and other map with targets (Integers) and the features (Double).
	 * @throws IOException
	 */
	public Map<Integer, Map<Integer, List<Double>>> read(String path) throws IOException {
		try {
			bfr = new BufferedReader(new FileReader(new File(path)));

			Map<Integer, Map<Integer, List<Double>>> myMap = new HashMap<>();
			Iterator<String> it = bfr.lines().iterator();
			it.next(); // Skiping first line
			int cont = 0;
			while (it.hasNext()) {
				String[] row = it.next().split(",");
				List<Double> features = new ArrayList<Double>();
				for (int i = 1; i < row.length; i++) {
					features.add(Double.valueOf(row[i]));
				}
				Map<Integer, List<Double>> map = new HashMap<>();
				map.put(Util.targets.get(row[0].toLowerCase()), features);
				myMap.put(cont, map);
				cont++;
			}
			return myMap;
		} finally {
			bfr.close();
			bfr = null;

		}
	}

	public static void main(String[] args) throws IOException {
		Map<Integer, Map<Integer, List<Double>>> map = LoadData.getInstance().read("../../../data/segmentation_2.test");

		List<List<Double>> values = new ArrayList<>();
		
		for (Map<Integer, List<Double>> yX: map.values()) {
			for (List<Double> example: yX.values()) {
//				System.out.println(example.stream().
//						map(Object::toString).
//						collect(Collectors.joining(" - ")));
				values.add(example);
			}
		}
//		System.out.println(map.keySet().size());
		Double[][] matValues = Util.normalize(values);
		for (Double[] features : matValues) {
			System.out.println(Arrays.stream(features).
					map(Object::toString).
					collect(Collectors.joining(" - ")));
		}

	}
}