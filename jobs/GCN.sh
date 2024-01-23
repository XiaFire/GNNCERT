cd GraphGuard

DATASET='MUTAG'
EPOCH_GG=100
SEED=42
FEATURES_METHOD='id'
DEGREE_AS_TAG=0
HASH_METHOD='md5'
GPU='0'
METHOD='structure'
ARCH='GCN'
for DATASET in 'MUTAG' 'PROTEINS_full' 'ENZYMES'; do
for NUM_GROUP in 30; do

NAME="new_GG_DS_${DATASET}_EPOCH_${EPOCH_GG}_SEED_${SEED}_FOLDN_3_FM_${FEATURES_METHOD}_DAT_${DEGREE_AS_TAG}_HM_${HASH_METHOD}_NG_${NUM_GROUP}" 
mkdir -p "./saved_model/${NAME}"
mkdir -p "./new_results/${NAME}"

MODEL_PATH="./saved_model/${NAME}/GG_idx0_${METHOD}_${ARCH}.pth"
LOG="./new_results/${NAME}/GG_idx0_${METHOD}_${ARCH}.csv"
if [ -e "${MODEL_PATH}" ]; then
      echo "Skipping execution for ${DATASET}_${NUM_GROUP}"
else
echo python3 graphguard_pyg.py --device "${GPU}" \
    --dataset "${DATASET}" \
    --seed="${SEED}" \
    --fold_n=3\
    --fold_idx=0 \
    --division-method=${METHOD}\
    --degree_as_tag "${DEGREE_AS_TAG}" \
    --features-method "${FEATURES_METHOD}" \
    --hash-method "${HASH_METHOD}"\
    --model_weight "${MODEL_PATH}"\
    --filename "${LOG}"\
    --epochs "${EPOCH_GG}"\
    --num_group "${NUM_GROUP}"\
    --arch "${ARCH}"
 python3 graphguard_pyg.py --device "${GPU}" \
    --dataset "${DATASET}" \
    --seed="${SEED}" \
    --fold_n=3\
    --fold_idx=0 \
    --division-method=${METHOD}\
    --degree_as_tag "${DEGREE_AS_TAG}" \
    --features-method "${FEATURES_METHOD}" \
    --hash-method "${HASH_METHOD}"\
    --model_weight "${MODEL_PATH}"\
    --filename "${LOG}"\
    --epochs "${EPOCH_GG}"\
    --num_group "${NUM_GROUP}"\
    --arch "${ARCH}"
fi
done
done