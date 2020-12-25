# '''
# import java.util.Arrays;
# import java.util.Random;
#
# import elki.clustering.ClusteringAlgorithmUtil;
# import elki.data.Cluster;
# import elki.data.Clustering;
# import elki.data.model.MedoidModel;
# import elki.database.ids.*;
# import elki.database.query.QueryBuilder;
# import elki.database.query.distance.DistanceQuery;
# import elki.database.relation.Relation;
# import elki.distance.Distance;
# import elki.logging.Logging;
# import elki.logging.progress.FiniteProgress;
# import elki.logging.statistics.DoubleStatistic;
# import elki.result.Metadata;
# import elki.utilities.Priority;
# import elki.utilities.documentation.Reference;
# import elki.utilities.exceptions.AbortException;
# import elki.utilities.random.RandomFactory;
# '''
#
# '''
#  * A faster variation of CLARANS, that can explore O(k) as many swaps at a
#  * similar cost by considering all medoids for each candidate non-medoid. Since
#  * this means sampling fewer non-medoids, we suggest to increase the subsampling
#  * rate slightly to get higher quality than CLARANS, at better runtime.
#
#  * Erich Schubert, Peter J. Rousseeuw<br>
#  * Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS
#  * Algorithms<br>
#  * Proc. 12th Int. Conf. Similarity Search and Applications (SISAP'2019)
#
#  * @has - - - Assignment
#  *
#  * @param <V> Vector type
# '''
# @Reference(authors = "Erich Schubert, Peter J. Rousseeuw", //
#     title = "Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms", //
#     booktitle = "Proc. 12th Int. Conf. Similarity Search and Applications (SISAP'2019)", //
#     url = "https://doi.org/10.1007/978-3-030-32047-8_16", //
#     bibkey = "DBLP:conf/sisap/SchubertR19")
# @Priority(Priority.IMPORTANT + 1)
# public class FastCLARANS<V> extends CLARANS<V> {
#   /**
#    * Class logger.
#    */
#   private static final Logging LOG = Logging.getLogger(FastCLARANS.class);
#
#   /**
#    * Constructor.
#    *
#    * @param distance Distance function to use
#    * @param k Number of clusters to produce
#    * @param numlocal Number of samples (restarts)
#    * @param maxneighbor Neighbor sampling rate (absolute or relative)
#    * @param random Random generator
#    */
#   public FastCLARANS(Distance<? super V> distance, int k, int numlocal, double maxneighbor, RandomFactory random) {
#     super(distance, k, numlocal, maxneighbor, random);
#   }
#
#   @Override
#   public Clustering<MedoidModel> run(Relation<V> relation) {
#     if(k * 2 >= relation.size()) {
#       // Random sampling of non-medoids will be slow for huge k
#       LOG.warning("A very large k was chosen. This implementation is not optimized for this case.");
#     }
#     DBIDs ids = relation.getDBIDs();
#     DistanceQuery<V> distQ = new QueryBuilder<>(relation, distance).distanceQuery();
#     final boolean metric = distance.isMetric();
#
#     // Number of retries, relative rate, or absolute count:
#     final int retries = (int) Math.ceil(maxneighbor < 1 ? maxneighbor * (ids.size() - k) : maxneighbor);
#     Random rnd = random.getSingleThreadedRandom();
#     // We will be using this to avoid sampling the same points twice.
#     ArrayModifiableDBIDs subsampler = DBIDUtil.newArray(ids);
#     DBIDArrayIter cand = subsampler.iter();
#
#     // Setup cluster assignment store
#     Assignment best = new Assignment(distQ, ids, k);
#     Assignment curr = new Assignment(distQ, ids, k);
#     double[] pcost = new double[k];
#
#     // 1. initialize
#     double bestscore = Double.POSITIVE_INFINITY;
#     FiniteProgress prog = LOG.isVerbose() ? new FiniteProgress("CLARANS sampling restarts", numlocal, LOG) : null;
#     for(int i = 0; i < numlocal; i++) {
#       // 2. choose random initial medoids
#       curr.medoids.clear().addDBIDs(DBIDUtil.randomSample(ids, k, rnd));
#       // Cost of initial solution:
#       double total = curr.assignToNearestCluster();
#       curr.computeRemovalCost(pcost);
#
#       // 3. Set j to 1.
#       int j = 1;
#       step: while(j < retries) {
#         // 4 part a. choose a random non-medoid (~ neighbor in G):
#         for(int r = 0; r < ids.size(); r++) {
#           // Fisher-Yates shuffle to avoid sampling the same points twice!
#           subsampler.swap(r, rnd.nextInt(ids.size() - r) + r); // Random point
#           cand.seek(r); // Random point
#           if(curr.nearest.doubleValue(cand) > 0) {
#             break; // Good: not a medoid.
#           }
#           // We may have many duplicate points
#           if(metric && curr.second.doubleValue(cand) == 0) {
#             ++j; // Cannot yield an improvement if we are metric.
#             continue step;
#           }
#           else if(!metric && !curr.medoids.contains(cand)) {
#             // Probably not a good candidate, but try nevertheless
#             break;
#           }
#           if(r >= 1000) {
#             throw new AbortException("Failed to choose a non-medoid in 1000 attempts. Choose k << N.");
#           }
#           // else: this must be the medoid.
#         }
#         // 5. check lower cost
#         double cost = curr.computeCostDifferential(cand, pcost);
#         if(!(cost < -1e-12 * total)) {
#           ++j; // 6. try again
#           continue;
#         }
#         total += cost; // cost is negative!
#         // Swap:
#         curr.performLastSwap(cand);
#         curr.computeRemovalCost(pcost);
#         j = 1;
#       }
#       if(LOG.isStatistics()) {
#         LOG.statistics(new DoubleStatistic(getClass().getName() + ".sample-" + i + ".cost", total));
#       }
#       // New best:
#       if(total < bestscore) {
#         // Swap:
#         Assignment tmp = curr;
#         curr = best;
#         best = tmp;
#         bestscore = total;
#       }
#       LOG.incrementProcessed(prog);
#     }
#     LOG.ensureCompleted(prog);
#     if(LOG.isStatistics()) {
#       LOG.statistics(new DoubleStatistic(getClass().getName() + ".final-cost", bestscore));
#     }
#
#     ArrayModifiableDBIDs[] clusters = ClusteringAlgorithmUtil.partitionsFromIntegerLabels(ids, best.assignment, k);
#     // Wrap result
#     Clustering<MedoidModel> result = new Clustering<>();
#     Metadata.of(result).setLongName("CLARANS Clustering");
#     for(DBIDArrayIter it = best.medoids.iter(); it.valid(); it.advance()) {
#       result.addToplevelCluster(new Cluster<>(clusters[it.getOffset()], new MedoidModel(DBIDUtil.deref(it))));
#     }
#     return result;
#   }
#
#   /**
#    * Assignment state.
#    *
#    * @author Erich Schubert
#    */
#   protected static class Assignment extends CLARANS.Assignment {
#     /**
#      * Array for storing the per-medoid loss.
#      */
#     double[] loss;
#
#     /**
#      * Last best medoid number
#      */
#     protected int lastbest;
#
#     /**
#      * Constructor.
#      *
#      * @param distQ Distance query
#      * @param ids IDs to process
#      * @param k Number of medoids
#      */
#     public Assignment(DistanceQuery<?> distQ, DBIDs ids, int k) {
#       super(distQ, ids, k);
#       loss = new double[k];
#     }
#
#     /**
#      * Precompute the costs of reassigning to the second closest medoid.
#      *
#      * @param pcost Output cost
#      */
#     protected void computeRemovalCost(double[] pcost) {
#       Arrays.fill(pcost, 0.);
#       for(DBIDIter j = ids.iter(); j.valid(); j.advance()) {
#         pcost[assignment.intValue(j)] += second.doubleValue(j) - nearest.doubleValue(j);
#       }
#     }
#
#     /**
#      * Compute the reassignment cost, for one swap.
#      *
#      * @param h Current object to swap with any medoid
#      * @param pcost Prior cost for removing
#      * @return Cost change
#      */
#     protected double computeCostDifferential(DBIDRef h, double[] pcost) {
#       System.arraycopy(pcost, 0, loss, 0, pcost.length);
#       double acc = 0.; // Cost accumulator for all
#       final int k = loss.length;
#       // Compute costs of reassigning other objects j:
#       for(DBIDIter j = ids.iter(); j.valid(); j.advance()) {
#         // distance(j, i) to nearest medoid
#         final double distcur = nearest.doubleValue(j);
#         // distance(j, k) to second nearest medoid
#         final double distsec = second.doubleValue(j);
#         // distance(j, h) to new medoid
#         final double dist_h = distQ.distance(h, j);
#         if(dist_h < distcur) {
#           acc += dist_h - distcur;
#           loss[assignment.intValue(j)] += distcur - distsec;
#         }
#         else if(dist_h < distsec) {
#           loss[assignment.intValue(j)] += dist_h - distsec;
#         }
#       }
#       // Add shared costs when processing the cost anyway
#       double min = loss[0] += acc;
#       lastbest = 0;
#       for(int i = 1; i < k; i++) {
#         if(loss[i] < min) {
#           min = loss[i] += acc;
#           lastbest = i;
#         }
#       }
#       return min;
#     }
#
#     /**
#      * Compute the reassignment cost, for one swap.
#      *
#      * @param h Current object to swap with the best medoid
#      */
#     protected void performLastSwap(DBIDRef h) {
#       // Update medoids of scratch copy.
#       medoids.set(lastbest, h);
#       // Compute costs of reassigning other objects j:
#       for(DBIDIter j = ids.iter(); j.valid(); j.advance()) {
#         if(DBIDUtil.equal(h, j)) {
#           recompute(j, lastbest, 0., -1, Double.POSITIVE_INFINITY);
#           continue;
#         }
#         // distance(j, i) to nearest medoid
#         final double distcur = nearest.doubleValue(j);
#         // distance(j, h) to new medoid
#         final double dist_h = distQ.distance(h, j);
#         // current assignment of j
#         final int jcur = assignment.intValue(j);
#         // Check if current medoid of j is removed:
#         if(jcur == lastbest) {
#           // distance(j, o) to second nearest / possible reassignment
#           final double distsec = second.doubleValue(j);
#           // Case 1b: j switches to new medoid, or to the second nearest:
#           if(dist_h < distsec) {
#             assignment.putInt(j, lastbest);
#             nearest.putDouble(j, dist_h);
#             second.putDouble(j, distsec);
#             secondid.putInt(j, jcur);
#           }
#           else {
#             // We have to recompute, because we do not know the true new second
#             // nearest.
#             recompute(j, lastbest, dist_h, jcur, distsec);
#           }
#         }
#         else if(dist_h < distcur) {
#           // Case 1c: j is closer to h than its current medoid
#           // and the current medoid is not removed (jcur != mnum).
#           // Second nearest is the previous assignment
#           assignment.putInt(j, lastbest);
#           nearest.putDouble(j, dist_h);
#           second.putDouble(j, distcur);
#           secondid.putInt(j, jcur);
#         }
#         else { // else Case 1a): j is closer to i than h and m, so no change.
#           final int jsec = secondid.intValue(j);
#           final double distsec = second.doubleValue(j);
#           // Second nearest is still valid.
#           if(jsec != lastbest && distsec <= dist_h) {
#             assignment.putInt(j, jcur);
#             nearest.putDouble(j, distcur);
#             secondid.putInt(j, jsec);
#             second.putDouble(j, distsec);
#           }
#           else {
#             recompute(j, jcur, distcur, lastbest, dist_h);
#           }
#         }
#       }
#     }
#   }
#
#   /**
#    * Parameterization class.
#    *
#    * @author Erich Schubert
#    */
#   public static class Par<V> extends CLARANS.Par<V> {
#     @Override
#     protected double defaultRate() {
#       return 2 * super.defaultRate();
#     }
#
#     @Override
#     public FastCLARANS<V> make() {
#       return new FastCLARANS<>(distance, k, numlocal, maxneighbor, random);
#     }
#   }
# }