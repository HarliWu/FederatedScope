import logging

from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import b64serializer
from federatedscope.core.workers.server import Server

from federatedscope.llm.offsite_tuning.utils import \
    generate_emulator_and_adapter

logger = logging.getLogger(__name__)


class OffsiteTuningServer(Server):
    """
    Server implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        compress_strategy = config.llm.offsite_tuning.strategy
        self.raw_model = model
        emulator_l = config.llm.offsite_tuning.emu_l
        emulator_r = config.llm.offsite_tuning.emu_r
        offsite_tuning_kwargs = config.llm.offsite_tuning.kwargs[0]
        logger.info('Server: Generating emulator and adapter...')
        adap_model = \
            generate_emulator_and_adapter(model,
                                          strategy=compress_strategy,
                                          emulator_l=emulator_l,
                                          emulator_r=emulator_r,
                                          **offsite_tuning_kwargs)
        super(OffsiteTuningServer,
              self).__init__(ID, state, config, data, adap_model, client_num,
                             total_round_num, device, strategy, **kwargs)

        # # Append a new model to the model list
        # self.model_num += 1
        # self.models.append(self.raw_model)

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        logger.info('Server: Converting emulator and adapter...')
        emulator_and_adapter = b64serializer(self._model, tool='dill')

        self.comm_manager.send(
            Message(msg_type='emulator_and_adapter',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    timestamp=self.cur_timestamp,
                    content=emulator_and_adapter))
        trigger_train_func(**kwargs_for_trigger_train_func)

    def eval(self):
        # Evaluate new emulator and adaptor
        super().eval()

        # Replace new emulator with original model
        self.raw_model.load_state_dict(self.model.state_dict(), strict=False)
        # dispatch the model to all clients
        # TODO: debug this part, not finished yet
        raw_model = b64serializer(self.raw_model, tool='dill')
        self.comm_manager.send(
            Message(msg_type='eval_plugin',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    timestamp=self.cur_timestamp,
                    content=raw_model))
        pass

    # TODO: merge plugin results and emulatopr results together
    def _merge_and_format_eval_results(self):
        return super()._merge_and_format_eval_results()
