import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.InvocationTargetException;

public class BloomFilterJoinV6 {
    private static Path scoreCachePath = new Path("/input/score.csv");
    private static Path studentCachePath = new Path("/input/student.csv");
    private static long[] primes = {4787, 7919};


    @FunctionalInterface
    public interface LineChecker {
        boolean check(String line);
    }

    public static class TokenizerMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
        private LongWritable longKey = new LongWritable();
        private Text attributes = new Text();
        private boolean[] scoreFilter = new boolean[7919];
        private boolean[] studentFilter = new boolean[7919];

        private LineChecker studentChecker = (String record) -> {
            int pivot = record.indexOf(",");
            return Integer.parseInt(record.substring(pivot + 1)) >= 1990;
        };

        private LineChecker scoreChecker = (String record) -> {
            String[] scores = record.split(",");
            for (int i = 0; i < 3; i++)
                if (Integer.parseInt(scores[i]) <= 80) return false;
            return true;
        };

        private void setupFilter(boolean[] filter, Path path, LineChecker checker) throws IOException, InterruptedException, InvocationTargetException {
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

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            long start = System.currentTimeMillis();
            try {
                setupFilter(scoreFilter, scoreCachePath, scoreChecker);
                setupFilter(studentFilter, studentCachePath, studentChecker);
            } catch (InvocationTargetException ex) {
                throw new IOException("Invocation failed!");
            }
            long timeSpent = System.currentTimeMillis() - start;
            System.out.println("Time spent forming BloomFilter was: " + timeSpent + " ms.");
        }

        public void map(LongWritable key, Text text, Context context) throws IOException, InterruptedException {
            String line = text.toString();
            int pivot = line.indexOf(",");
            long studentKey = Long.parseLong(line.substring(0, pivot));

            FileSplit split = (FileSplit) context.getInputSplit();
            String inputFile = split.getPath().getName();
            String tag;

            if (inputFile.equals("score.csv")) {
                for (long prime : primes)
                    if (!studentFilter[(int) (studentKey % prime)]) return;
                tag = "X";
            } else if (inputFile.equals("student.csv")) {
                for (long prime : primes)
                    if (!scoreFilter[(int) (studentKey % prime)]) return;
                tag = "Y";
            } else {
                return;
            }

            attributes.set(tag + "," + line.substring(pivot + 1));
            longKey.set(studentKey);

            context.write(longKey, attributes);
        }
    }

    public static class ResultCombiner
            extends Reducer<LongWritable, Text, LongWritable, Text> {
        private Text result = new Text();
        private StringBuilder sb = new StringBuilder();

        public void reduce(LongWritable key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            String[] records = new String[2];
            int ind = 0;

            for (Text val : values)
                records[ind++] = val.toString();

            if (ind != 2) return;
            String[] scoreData;
            String[] studentData;
            sb.setLength(0);

            if (records[0].charAt(0) == 'X') {
                scoreData = records[0].substring(2).split(",");
                studentData = records[1].substring(2).split(",");

                if (Integer.parseInt(studentData[1]) <= 1989) return;
                for (int i = 0; i < 3; i++)
                    if (Integer.parseInt(scoreData[i]) <= 80) return;

                sb.append(records[0].substring(2));
                sb.append(",");
                sb.append(records[1].substring(2));
            } else if (records[0].charAt(0) == 'Y') {
                scoreData = records[1].substring(2).split(",");
                studentData = records[0].substring(2).split(",");

                if (Integer.parseInt(studentData[1]) <= 1989) return;
                for (int i = 0; i < 3; i++)
                    if (Integer.parseInt(scoreData[i]) <= 80) return;

                sb.append(records[1].substring(2));
                sb.append(",");
                sb.append(records[0].substring(2));
            } else {
                sb.append("ERROR: " + ind + " " + records[0]);
            }

            result.set(sb.toString());
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "simple reduce-side join");
        job.setJarByClass(BloomFilterJoinV6.class);
        job.setMapperClass(BloomFilterJoinV6.TokenizerMapper.class);
//        job.setCombinerClass(BloomFilterJoinV6.ResultCombiner.class);
        job.setReducerClass(BloomFilterJoinV6.ResultCombiner.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        DistributedCache.addCacheFile(scoreCachePath.toUri(), job.getConfiguration());
        DistributedCache.addCacheFile(studentCachePath.toUri(), job.getConfiguration());

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}