import org.apache.hadoop.fs.Path;

import java.io.*;
import java.lang.reflect.InvocationTargetException;

public class BloomFilterConstructor {
    private static Path scoreCachePath = new Path("/input/score.csv");
    private static Path studentCachePath = new Path("/input/student.csv");
    private static int MAX_PRIME = 7561009;
    private static long[] primes = {7561009, 3123823, 2414417, 1299811, 1174319, 1087241, 954067, 915527, 778993,
            697019, 670763, 506281, 579757};
    private boolean[] scoreFilter = new boolean[MAX_PRIME];
    private boolean[] studentFilter = new boolean[MAX_PRIME];

    private BloomFilterJoinV7.LineChecker studentChecker = (String record) -> {
        int pivot = record.indexOf(",");
        return Integer.parseInt(record.substring(pivot + 1)) >= 1990;
    };

    private BloomFilterJoinV7.LineChecker scoreChecker = (String record) -> {
        String[] scores = record.split(",");
        for (int i = 0; i < 3; i++)
            if (Integer.parseInt(scores[i]) <= 80) return false;
        return true;
    };

    private void setupFilter(boolean[] filter, Path path, BloomFilterJoinV7.LineChecker checker) throws IOException, InterruptedException, InvocationTargetException {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream(
                                path.getName()
                        )
                ));
        String line;
        while ((line = br.readLine()) != null) {
            int pivot = line.indexOf(",");
            long key = Long.parseLong(line.substring(0, pivot));
            if (checker.check(line.substring(pivot + 1)))
                for (long prime : primes)
                    filter[(int) (key % prime)] = true;
        }
        br.close();
    }

    public void construct() throws IOException, InterruptedException {
        try {
            setupFilter(scoreFilter, scoreCachePath, scoreChecker);
            setupFilter(studentFilter, studentCachePath, studentChecker);
        } catch (InvocationTargetException ex) {
            throw new IOException("Invocation failed!");
        }
    }

    public void writeScoreFilter(String path) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        for (int i = 0; i < MAX_PRIME; i++)
            if (scoreFilter[i]) writer.write(1);
            else writer.write(0);
        writer.close();
    }

    public void writeStudentFilter(String path) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        for (int i = 0; i < MAX_PRIME; i++)
            if (scoreFilter[i]) writer.write(1);
            else writer.write(0);
        writer.close();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        BloomFilterConstructor bc = new BloomFilterConstructor();
        bc.construct();
        bc.writeStudentFilter("studentFilter.txt");
        bc.writeScoreFilter("scoreFilter.txt");
    }
}