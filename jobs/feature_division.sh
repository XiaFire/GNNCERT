cd GraphGuard

DATASET='MUTAG'
EPOCH_GG=1000
SEED=42
FEATURES_METHOD='id'
DEGREE_AS_TAG=0
HASH_METHOD='md5'
GPU='0'

for DATASET in 'MUTAG' 'PROTEINS_full' 'DD' 'ENZYMES' 'NCI1' 'COLLAB' 'REDDITBINARY' 'DBLP_v1'; do
for NUM_GROUP in 10 30 50; do

NAME="GG_DS_${DATASET}_EPOCH_${EPOCH_GG}_SEED_${SEED}_FOLDN_3_FM_${FEATURES_METHOD}_DAT_${DEGREE_AS_TAG}_HM_${HASH_METHOD}_NG_${NUM_GROUP}" 
mkdir -p "./saved_model/${NAME}"
mkdir -p "./new_results/${NAME}"

MODEL_PATH="./saved_model/${NAME}/GG_idx0_feature.pth"
LOG="./new_results/${NAME}/GG_idx0_feature.csv"
if [ -e "${MODEL_PATH}" ]; then
      echo "Skipping execution for ${DATASET}_${NUM_GROUP}"
else

python3 main_graphguard.py --device "${GPU}" \
    --dataset "${DATASET}" \
    --seed="${SEED}" \
    --fold_n=3\
    --fold_idx=0 \
    --division-method=feature \
    --degree_as_tag "${DEGREE_AS_TAG}" \
    --features-method "${FEATURES_METHOD}" \
    --hash-method "${HASH_METHOD}"\
    --model_weight "${MODEL_PATH}"\
    --filename "${LOG}"\
    --epochs "${EPOCH_GG}"\
    --num_group "${NUM_GROUP}"

fi
done
done

