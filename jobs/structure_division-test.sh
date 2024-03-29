cd GraphGuard

DIV_METHOD='structure'
EPOCH_GG=1000
SEED=42
FEATURES_METHOD='id'
DEGREE_AS_TAG=0
HASH_METHOD='md5'
GPU='0'

for DATASET in 'MUTAG' 'PROTEINS_full' 'ENZYMES'; do
for NUM_GROUP in 30; do
for HASH_METHOD in 'md5'; do
NAME="1timeGG_DS_${DATASET}_EPOCH_${EPOCH_GG}_SEED_${SEED}_FOLDN_3_FM_${FEATURES_METHOD}_DAT_${DEGREE_AS_TAG}_HM_${HASH_METHOD}_NG_${NUM_GROUP}" 
mkdir -p "./saved_model/${NAME}"
mkdir -p "./new_results/${NAME}"
MODEL_PATH="./saved_model/${NAME}/GG_idx0_${DIV_METHOD}.pth"
LOG="./new_results/${NAME}/GG_idx0_${DIV_METHOD}.csv"

python3 test.py --device "${GPU}" \
    --dataset "${DATASET}" \
    --seed="${SEED}" \
    --fold_n=3\
    --fold_idx=0 \
    --hash-method="${HASH_METHOD}"\
    --degree_as_tag "${DEGREE_AS_TAG}"\
    --features-method "${FEATURES_METHOD}" \
    --division-method "${DIV_METHOD}" \
    --defense_method "GG"\
    --model_weight "${MODEL_PATH}"\
    --num_group "${NUM_GROUP}"\
    --filename "${LOG}_${test}"

done
done
done
