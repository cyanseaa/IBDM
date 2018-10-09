import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import java.io.IOException;
import java.util.*;

import static org.apache.commons.math3.util.FastMath.min;

public class MapRed {
    static final int q = 2;
    static final int chunkSize = 5000;
    private static float epsilon = (float) 1e-9;


    public static class QGramFinder extends Mapper<LongWritable, Text, Text, Text> {
        private Text wikipediaEntry = new Text();
        private Text oval = new Text();

        /**
         * Finds qgrams in entry and writes tuples (entry,qgram,fileTag,qgramAmount)
         *
         * @param okey    not used but required by Hadoop
         * @param text
         * @param context
         * @throws InterruptedException
         * @throws IOException
         */
        @Override
        public void map(LongWritable okey, Text text, Context context) throws InterruptedException, IOException {
            String line = text.toString().trim();
            if (line.length() < q) return;
            Map<String, Integer> qgrams = new HashMap<>();
            for (int i = q - 1; i < line.length(); i++) {
                String qgram = line.substring(i - q + 1, i + 1);
                qgrams.put(qgram, qgrams.getOrDefault(qgram, 0) + 1);
            }
            wikipediaEntry.set(line);

            FileSplit split = (FileSplit) context.getInputSplit();
            String inputFile = split.getPath().getName();
            String tag = "X";
            if (inputFile.equals("s2.txt")) tag = "Y";

            for (Map.Entry qgram : qgrams.entrySet()) {
                oval.set(qgram.getKey() + " " + tag + " " + qgram.getValue());
                context.write(wikipediaEntry, oval);
            }
        }
    }

    public static class CountByQram extends Mapper<Text, Text, Text, Text> {
        private Text okey = new Text();
        private Text oval = new Text();

        /**
         * Swaps key (entry) and qgram to form entries by qgram.
         *
         * @param ikey
         * @param ival    triple: fileTag qgram count
         * @param context
         * @throws InterruptedException
         * @throws IOException
         */
        @Override
        public void map(Text ikey, Text ival, Context context) throws InterruptedException, IOException {
            StringTokenizer tokenizer = new StringTokenizer(ival.toString());
            okey.set(tokenizer.nextToken());
            String tag = tokenizer.nextToken();
            int count = Integer.parseInt(tokenizer.nextToken());
            oval.set(tag + " " + ikey.toString() + " " + count);
            context.write(okey, oval);
        }
    }

    public static class SplitToChunks extends Reducer<Text, Text, Text, Text> {
        private Text okey = new Text();
        private Text ovalue = new Text();

        /**
         * Splits list of triples (fileTag,entry,qgramcount) to chunks.
         */
        @Override
        public void reduce(Text ikey, Iterable<Text> ival, Context context) throws InterruptedException, IOException {
            int counterX = 0, counterY = 0;
            ArrayList<String> chunks, otherChunks;
            ArrayList<String> xChunks = new ArrayList<>();
            ArrayList<String> yChunks = new ArrayList<>();


            for (Text pair : ival) {
                String cur = pair.toString();
                char tag = cur.charAt(0);
                if (tag == 'X') {
                    chunks = xChunks;
                    otherChunks = yChunks;
                    if (counterX % chunkSize == 0) chunks.add(cur);
                    else chunks.set(chunks.size() - 1, cur + " " + chunks.get(chunks.size() - 1));
                    counterX++;
                } else {
                    chunks = yChunks;
                    otherChunks = xChunks;
                    if (counterY % chunkSize == 0) chunks.add(cur);
                    else chunks.set(chunks.size() - 1, cur + " " + chunks.get(chunks.size() - 1));
                    counterY++;
                }

                for (String chunk : otherChunks) {
                    okey.set(UUID.randomUUID().toString());
                    ovalue.set(cur + " " + chunk);
                    context.write(okey, ovalue);
                }
            }
        }
    }

    public static class PairCollector extends Mapper<Text, Text, Text, IntWritable> {
        private IntWritable oval = new IntWritable();
        private Text okey = new Text();

        /**
         * Pairs elements in the input list with first element forming triples (e_0,e_i,amount of common qgram)
         *
         * @param ikey    artificial key, not used
         * @param ival    list of pair (entry,fileTag,amount) where entry has amount many time the same qgram.
         * @param context
         * @throws InterruptedException
         * @throws IOException
         */
        @Override
        public void map(Text ikey, Text ival, Context context) throws InterruptedException, IOException {
            String list = ival.toString();

            StringTokenizer tokenizer = new StringTokenizer(list);
            int firstCount;
            String firstMatch, firstfileTag;
            boolean firstIsX;

            if (tokenizer.hasMoreTokens()) {
                firstfileTag = tokenizer.nextToken();
                firstIsX = firstfileTag.equals("X");
                if (tokenizer.hasMoreTokens()) {
                    firstMatch = tokenizer.nextToken();
                    if (tokenizer.hasMoreTokens()) {
                        firstCount = Integer.parseInt(tokenizer.nextToken());

                        while (tokenizer.hasMoreTokens()) {
                            String tag = tokenizer.nextToken();
                            if (tokenizer.hasMoreTokens()) {
                                String k = tokenizer.nextToken();
                                if (tokenizer.hasMoreTokens()) {
                                    int v = Integer.parseInt(tokenizer.nextToken());

                                    //Should be unnecessary here due to chunking system!
                                    if (firstfileTag.equals(tag)) continue;

                                    if (firstIsX) okey.set(firstMatch + " " + k);
                                    else okey.set(k + " " + firstMatch);
                                    oval.set(min(firstCount, v));
                                    context.write(okey, oval);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    //Outputting JaccardDistances for sanity checking, not really necessary later on!
    public static class JaccardSimilarityReducer extends Reducer<Text, IntWritable, Text, FloatWritable> {
        private FloatWritable oval = new FloatWritable();

        /**
         * Computes estimation for Jaccard distance of two entries based on how amny times each qgram was met in estimation.
         * Can only overestimate Jaccard distance by not including everything in common.
         *
         * @param ikey        pair (entry1,entry2)
         * @param numInCommon number of each qgram in common.
         * @param context
         * @throws InterruptedException
         * @throws IOException
         */
        @Override
        public void reduce(Text ikey, Iterable<IntWritable> numInCommon, Context context) throws InterruptedException, IOException {
            int common = 0;
            for (IntWritable amount : numInCommon)
                common += amount.get();
            StringTokenizer tokenizer = new StringTokenizer(ikey.toString());
            int total = tokenizer.nextToken().length() + tokenizer.nextToken().length() - 2 * (q - 1);
            float distance = 1 - (float) common / (float) (total - common);

            if (distance > epsilon && distance < 0.15 + epsilon) {
                oval.set(distance);
                context.write(ikey, oval);
            }
        }
    }

    public static class BruteForceSimilarity extends Reducer<Text, FloatWritable, Text, FloatWritable> {
        private FloatWritable oval = new FloatWritable();

        /**
         * Brute forces to verify earlier computations.
         *
         * @param ikey             pair (entry1, entry2) of wto wikipedia entries that were found similar in filtering.
         * @param approxSimilarity approxmiated similarity between two entries, not used.
         * @param context
         * @throws InterruptedException
         * @throws IOException
         */
        @Override
        public void reduce(Text ikey, Iterable<FloatWritable> approxSimilarity, Context context) throws InterruptedException, IOException {
            // Should consider multiples of qgrams also earlier!
            Map<String, Integer> qgramsInFirst = new HashMap<>();
            Map<String, Integer> qgramsInSecond = new HashMap<>();
            StringTokenizer reader = new StringTokenizer(ikey.toString());

            String entry1 = reader.nextToken();
            String entry2 = reader.nextToken();

            for (int i = q - 1; i < entry1.length(); i++) {
                String qgram = entry1.substring(i - q + 1, i + 1);
                qgramsInFirst.put(qgram, qgramsInFirst.getOrDefault(qgram, 0) + 1);
            }

            for (int i = q - 1; i < entry2.length(); i++) {
                String qgram = entry2.substring(i - q + 1, i + 1);
                qgramsInSecond.put(qgram, qgramsInSecond.getOrDefault(qgram, 0) + 1);
            }

            int total = entry1.length() + entry2.length() - 2 * (q - 1);
            int common = 0;

            for (Map.Entry<String, Integer> entry : qgramsInFirst.entrySet())
                if (qgramsInSecond.containsKey(entry.getKey()))
                    common += min(entry.getValue(), qgramsInSecond.get(entry.getKey()));

            float distance = 1 - (float) common / (float) (total - common);
            oval.set(distance);

            context.write(ikey, approxSimilarity.iterator().next());
        }

    }

    public static void filter_step1(Configuration conf, String inputPath, String outputPath) throws Exception {
        Job job1 = Job.getInstance(conf, "Filter phase 1");
        job1.setJarByClass(MapRed.class);
        job1.setMapperClass(MapRed.QGramFinder.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        job1.setOutputFormatClass(SequenceFileOutputFormat.class);

        FileInputFormat.addInputPath(job1, new Path(inputPath));
        FileOutputFormat.setOutputPath(job1, new Path(outputPath, "intermediate1"));
        if (!job1.waitForCompletion(true)) {
            System.exit(1);
        }
    }

    public static void filter_step2(Configuration conf, String outputPath) throws Exception {
        Job job2 = Job.getInstance(conf, "Filter phase 2");
        job2.setJarByClass(MapRed.class);
        job2.setMapperClass(MapRed.CountByQram.class);
        job2.setReducerClass(MapRed.SplitToChunks.class);

        job2.setInputFormatClass(SequenceFileInputFormat.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        job2.setOutputFormatClass(SequenceFileOutputFormat.class);

        FileInputFormat.addInputPath(job2, new Path(outputPath, "intermediate1"));
        FileOutputFormat.setOutputPath(job2, new Path(outputPath, "intermediate2"));
        if (!job2.waitForCompletion(true)) {
            System.exit(1);
        }
    }

    public static void filter_step3(Configuration conf, String outputPath) throws Exception {
        Job job3 = Job.getInstance(conf, "Filter phase 3");
        job3.setJarByClass(MapRed.class);
        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(IntWritable.class);
        job3.setMapperClass(MapRed.PairCollector.class);
        job3.setReducerClass(MapRed.JaccardSimilarityReducer.class);

        job3.setInputFormatClass(SequenceFileInputFormat.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(FloatWritable.class);
//        job3.setOutputFormatClass(SequenceFileOutputFormat.class);


        FileInputFormat.addInputPath(job3, new Path(outputPath, "intermediate2"));
        FileOutputFormat.setOutputPath(job3, new Path(outputPath, "filter"));
        if (!job3.waitForCompletion(true)) {
            System.exit(1);
        }
    }

    public static void verify(Configuration conf, String outputPath) throws Exception {
        Job job4 = Job.getInstance(conf, "Verification phase");
        job4.setJarByClass(MapRed.class);
        job4.setReducerClass(MapRed.BruteForceSimilarity.class);

        job4.setInputFormatClass(SequenceFileInputFormat.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(FloatWritable.class);

        FileInputFormat.addInputPath(job4, new Path(outputPath, "filter"));
        FileOutputFormat.setOutputPath(job4, new Path(outputPath, "verified"));
        if (!job4.waitForCompletion(true)) {
            System.exit(1);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String inputPath = args[0];
        String outputPath = args[1];
        filter_step1(conf, inputPath, outputPath);
        filter_step2(conf, outputPath);
        filter_step3(conf, outputPath);
//        verify(conf, outputPath);
    }
}