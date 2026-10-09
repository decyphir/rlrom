from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from typing import Literal, Sequence, Tuple, Union, get_args

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import Action, Scenario, CustomScenario, parseTime
from simglucose.simulation.sim_engine import SimObj


TIME_TYPE = Union[int, float, timedelta, datetime]

# See Name column in params/vpatient_params.csv
PATIENT_TYPE = Literal[
    "adolescent#001", "adolescent#002",  "adolescent#003", "adolescent#004",  "adolescent#005", "adolescent#006", "adolescent#007", "adolescent#008", "adolescent#009", "adolescent#010",
    "adult#001", "adult#002",  "adult#003", "adult#004",  "adult#005", "adult#006", "adult#007", "adult#008", "adult#009", "adult#010",
    "child#001", "child#002",  "child#003", "child#004",  "child#005", "child#006", "child#007", "child#008", "child#009", "child#010"
]
PATIENT_NAMES = get_args(PATIENT_TYPE)


# See Name column in params/pump_params.csv
PUMP_TYPE = Literal["Cozmo", "Insulet"]

# See Name column in params/sensor_params.csv
SENSOR_TYPE = Literal["Dexcom", "GuardianRT", "Navigator"]
# specify start_time as the beginning of today
START_TIME = datetime.combine(datetime.now().date(), datetime.min.time())
# Specify results saving path
RESULT_PATH = './out'


class GenerativeScenario(Scenario):
    def __init__(self, start_time: datetime,
                 scen_gen_fun: Callable[[], Generator[Tuple[TIME_TYPE, Action], None, None]]):
        '''
        scen_gen - a generator function yielding tuples (time, action), where time is a datetime or
                   timedelta or double, action is a namedtuple defined by
                   scenario.Action. When time is a timedelta, it is
                   interpreted as the time of start_time + time. Time in double
                   type is interpreted as time in timedelta with unit of hours
        '''
        Scenario.__init__(self, start_time=start_time)
        self._prev_itvl = (None, start_time)
        self._scen_gen_fun = scen_gen_fun
        self.reset()

    def get_action(self, t):
        assert t >= self._prev_itvl[0]
        try:
            raise NotImplementedError("TODO: Get meal size at a timepoint give the possibly infinite iterable.")
        except StopIteration:
            return Action(meal=0)  # Return dummy meal after reaching the end of the iterable

    def reset(self):
        self._scen_iter = self._scen_gen_fun()  # Create a new generator


def build_sim_obj(
        meals: Sequence[Tuple[TIME_TYPE, float]],
        patient_name: PATIENT_TYPE,
        patient_init_state = None,
        patient_random_init_bg: bool = False,
        patient_seed = None,
        patient_t0: float = 0.0,
        sensor_name: SENSOR_TYPE = "Dexcom",
        sensor_seed: int = 1,
        pump_name: PUMP_TYPE = "Insulet") -> SimObj:
    # Create a simulation environment
    patient = T1DPatient.withName(
        name=patient_name,
        init_state=patient_init_state,
        random_init_bg=patient_random_init_bg,
        seed=patient_seed,
        t0=patient_t0)
    sensor = CGMSensor.withName(sensor_name, seed=sensor_seed)
    pump = InsulinPump.withName(pump_name)

    # Custom scenario is a list of tuples (time, meal_size)
    scenario = CustomScenario(start_time=START_TIME, scenario=meals)
    env = T1DSimEnv(patient, sensor, pump, scenario)

    # Create a controller
    controller = BBController()

    # Put them together to create a simulation object
    return SimObj(env, controller, timedelta(days=1), animate=False, path=RESULT_PATH)
