import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public class DistributedCacheJoin {
    private static Path scoreCachePath = new Path("/problem2/score.csv");


    public static class TokenizerMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
        private LongWritable longKey = new LongWritable();
        private Text attributes = new Text();
        private Map<Long, String> scoreTable = new HashMap<>();
        private StringBuilder sb = new StringBuilder();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(
                            scoreCachePath.getName()
                    )
            ));
            String line;
            while ((line = br.readLine()) != null) {
                int pivot = line.indexOf(",");
                long key = Long.parseLong(line.substring(0, pivot));
                scoreTable.put(key, line.substring(pivot + 1));
            }
            br.close();
        }

        public void map(LongWritable key, Text text, Context context) throws IOException, InterruptedException {
            String line = text.toString();
            int pivot = line.indexOf(",");
            long studentKey = Long.parseLong(line.substring(0, pivot));
            if (!scoreTable.containsKey(studentKey)) return;

            String[] studentRecord = line.substring(pivot + 1).split(",");
            if (Integer.parseInt(studentRecord[1]) < 1990) return;
            String scoreRecord = scoreTable.get(studentKey);
            String[] scores = scoreRecord.split(",");

            for (int i = 0; i < 3; i++)
                if (Integer.parseInt(scores[i]) <= 80) return;

            sb.setLength(0);
            sb.append(line);
            sb.append(",");
            sb.append(scoreRecord);
            attributes.set(sb.toString());
            longKey.set(studentKey);

            context.write(longKey, attributes);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "simple reduce-side join");
        job.setJarByClass(DistributedCacheJoin.class);
        job.setMapperClass(DistributedCacheJoin.TokenizerMapper.class);
//        job.setCombinerClass(DistributedCacheJoin.ResultCombiner.class);
        job.setReducerClass(Reducer.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        //Add score file to local cache, assumes it was added to hdfs with
        //hadoop fs -mkdir /problem2
        //hadoop fs -copyFromLocal score.csv /problem2/score.csv
        DistributedCache.addCacheFile(scoreCachePath.toUri(), job.getConfiguration());

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
