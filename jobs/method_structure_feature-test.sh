cd GraphGuard

DIV_METHOD='all'
DATASET='MUTAG'
EPOCH_GG=2000
SEED=42
FEATURES_METHOD='id'
DEGREE_AS_TAG=0
HASH_METHOD='md5'
GPU='0'

for DATASET in  'MUTAG' 'PROTEINS_full' 'ENZYMES'; do
for NUM_GROUP in 30; do
NAME="GG_DS_${DATASET}_EPOCH_${EPOCH_GG}_SEED_${SEED}_FOLDN_3_FM_${FEATURES_METHOD}_DAT_${DEGREE_AS_TAG}_HM_${HASH_METHOD}_NG_${NUM_GROUP}" 
mkdir -p "./saved_model/${NAME}"
mkdir -p "./new_results/${NAME}"
MODEL_PATH="./saved_model/${NAME}/GG_idx0_${DIV_METHOD}.pth"
LOG="./new_results/${NAME}/GG_idx0_${DIV_METHOD}.csv"
if [ ! -e "${MODEL_PATH}" ]; then
      echo "$MODEL_PATH"
      echo "Skipping execution for ${DATASET}_${NUM_GROUP}"
else
python3 -u test.py --device "${GPU}" \
    --dataset "${DATASET}" \
    --seed="${SEED}" \
    --fold_n=3\
    --fold_idx=0 \
    --degree_as_tag "${DEGREE_AS_TAG}"\
    --features-method "${FEATURES_METHOD}" \
    --division-method "${DIV_METHOD}" \
    --hash-method "${HASH_METHOD}"\
    --defense_method "GG"\
    --model_weight "${MODEL_PATH}"\
    --num_group "${NUM_GROUP}"\
    --filename "${LOG}"
fi
done
done
