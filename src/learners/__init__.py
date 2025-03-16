from .q_learner import QLearner
from .qatten_learner import QattenLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qtran_transformation_learner import QLearner as QTranTRANSFORMATIONLearner
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .BDBE_CDI_IDI import QPLEX_curiosity_vdn_Learner_cds

from .fast_q_learner import fast_QLearner

from .BDBE_episodic_CDI_IDI import QPLEX_curiosity_vdn_Learner





REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qtran_transformation_learner"] = QTranTRANSFORMATIONLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["BDBE_CDI_IDI"] = QPLEX_curiosity_vdn_Learner_cds


REGISTRY["fast_QLearner"] = fast_QLearner
REGISTRY['BDBE_episodic_CDI_IDI']=QPLEX_curiosity_vdn_Learner




