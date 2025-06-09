import abc
import datetime
import functools
import json
import math
import os
import random
import re
import threading
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from typing_extensions import override

from jinja2 import Template
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components.agent import memory_component
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import clock, entity, entity_component, logging
from concordia.utils import (
    concurrency,
    helper_functions,
    measurements as measurements_lib,
)


CUSTOM_DEBUG_MODE = bool(os.environ.get("CUSTOM_DEBUG_MODE", False))
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DEFAULT_TEMPERATURE = 0.4
GENERATION_PARAMS = {
    "Expert": {
        entity.OutputType.FREE: {
            "temperature": 0.4,
            "max_tokens": int(1024 * 2),
        },
        entity.OutputType.CHOICE: {
            "temperature": 0.0,
            "max_tokens": int(1024 * 2),
        },
        entity.OutputType.FLOAT: {
            "temperature": 0.0,
            "max_tokens": int(1024 * 2),
        },
    },
    "Router": {
        "temperature": 0.0,
        "max_tokens": int(1024 * 2),
    },
    "ActionAggregator": {
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": int(1024 * 2),
    },
    "EmotionSynthesizer": {
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": int(1024 * 2),
    },
    "MemorySummary": {
        "temperature": 0.2,
        "max_tokens": int(1024 * 2),
    },
    "PlayersProfiles": {
        "temperature": 0.2,
        "max_tokens": int(1024 * 2),
    },
    "PlayersOverarchingGoal": {
        "temperature": 0.2,
        "max_tokens": int(1024 * 2),
    },
    "MixtureOfExpertsSummary": {
        "temperature": 0.2,
        "max_tokens": int(1024 * 2),
    },
    # RecentObservations is calculated automatically
}

AGGREGATOR_TYPE = "Aggregator"  # Aggregator or Ranker
USE_EMOTION_SYNTHESIZER = False
DEFAULT_TOP_K = 1
USE_DYNAMIC_TOP_K = False


# Estimation of number of model calls per action:
# 1. Components:
#   - 1 for MemorySummary
#   - (almost always) 0 for Recent Observations
# 2. Router: 1 call per action
# 3. Experts: top_k calls per action
# 4. Aggregator: 1 call per action if top_k > 1 and experts not agreed
# 5. Emotion synthesizer: 1 call per action if used
DYNAMIC_TOP_K_MAPPING = {
    1: None,
    3: [
        {
            "from": 0,
            "to": 100,
            "top_k": 3,
        },  # 100 * ( (1 + 0) + 1 + 3 + 1 + 1) = 100 * 7 = 700
        {
            "from": 100,
            "to": float("inf"),
            "top_k": 1,
        },  # 1 * ( (1 + 0) + 1 + 1 + 0 + 1) = 1 * 4 = 4 -> 4 model calls per action
    ],
    5: [
        {
            "from": 0,
            "to": 75,
            "top_k": 5,
        },  # 75 * ( (1 + 0) + 1 + 5 + 1 + 1) = 75 * 9 = 675
        {
            "from": 75,
            "to": 100,
            "top_k": 3,
        },  # 25 * ( (1 + 0) + 1 + 3 + 1 + 1) = 25 * 7 = 175 -> aggregated 850
        {
            "from": 100,
            "to": float("inf"),
            "top_k": 1,
        },  # 1 * ( (1 + 0) + 1 + 1 + 0 + 1) = 1 * 4 = 4 -> 4 model calls per action
    ],
}


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent.

    Args:
      config: The agent config to use.
      model: The language model to use.
      memory: The agent's memory object.
      clock: The clock to use.
      update_time_interval: Agent calls update every time this interval passes.

    Returns:
      An agent.
    """
    del update_time_interval
    if not config.extras.get("main_character", False):
        raise ValueError(
            "This function is meant for a main character "
            "but it was called on a supporting character."
        )

    agent_name = config.name
    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
    memory_component_name = (
        agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
    )
    measurements = measurements_lib.Measurements()

    components_of_agent = {}
    components_of_agent[memory_component_name] = (
        agent_components.memory_component.MemoryComponent(raw_memory)
    )

    observer_label = "Observer"
    observer = Observer(
        pre_act_key=observer_label,
        memory_component_name=memory_component_name,
    )
    components_of_agent[observer_label] = observer

    time_display_label = "Current time"
    time_display = agent_components.report_function.ReportFunction(
        function=clock.current_time_interval_str,
        pre_act_key=time_display_label,
        logging_channel=measurements.get_channel(time_display_label).on_next,
    )
    components_of_agent[time_display_label] = time_display

    goal_label = None
    if config.goal:
        goal_label = f"{agent_name}'s Overarching Goals"
        overarching_goal = PlayersOverarchingGoals(
            model=model,
            initial_goal=config.goal,
            pre_act_key=goal_label,
            logging_channel=measurements.get_channel(goal_label).on_next,
        )
        components_of_agent[goal_label] = overarching_goal

    conversation_history_label = "Conversation History"
    conversation_history = ConversationHistory(
        memory_component_name=memory_component_name,
        pre_act_key=conversation_history_label,
    )
    components_of_agent[conversation_history_label] = conversation_history

    memory_summary_label = f"{agent_name}'s Memory Summary"
    memory_summary = MemorySummary(
        model=model,
        current_time_component_name=time_display_label,
        memory_component_name=memory_component_name,
        conversation_history_component_name=conversation_history_label,
        pre_act_key=memory_summary_label,
        logging_channel=measurements.get_channel(memory_summary_label).on_next,
    )
    components_of_agent[memory_summary_label] = memory_summary

    players_profiles_label = "Profiles of Other Players"
    players_profiles = PlayersProfiles(
        model=model,
        current_time_component_name=time_display_label,
        memory_summary_component_name=memory_summary_label,
        memory_component_name=memory_component_name,
        conversation_history_component_name=conversation_history_label,
        pre_act_key=players_profiles_label,
        logging_channel=measurements.get_channel(players_profiles_label).on_next,
    )
    components_of_agent[players_profiles_label] = players_profiles

    recent_observations_label = f"{agent_name}'s Recent Observations"
    recent_observations = RecentObservations(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=datetime.timedelta(hours=24),
        timeframe_delta_until=datetime.timedelta(hours=0),
        max_allowed_tokens=512,
        memory_component_name=memory_component_name,
        pre_act_key=recent_observations_label,
    )
    components_of_agent[recent_observations_label] = recent_observations

    # mixture_of_experts_summary_label = "Mixture-of-Experts Summary"
    # mixture_of_experts_summary = MixtureOfExpertsSummary(
    # model=model,
    # current_time_component_name=time_display_label,
    # memory_summary_component_name=memory_summary_label,
    # memory_component_name=memory_component_name,
    # pre_act_key=mixture_of_experts_summary_label,
    # logging_channel=measurements.get_channel(
    # mixture_of_experts_summary_label
    # ).on_next,
    # )
    # components_of_agent[mixture_of_experts_summary_label] = mixture_of_experts_summary

    expert_components_keys = [
        time_display_label,
        goal_label,
        memory_summary_label,
        players_profiles_label,
        conversation_history_label,
        recent_observations_label,
    ]
    expert_components_keys = list(
        filter(lambda x: x is not None, expert_components_keys)
    )
    scenarios = [
        Scenario(
            name=scenario["name"],
            description=scenario["description"],
            experts=[
                Expert(
                    pre_act_key=expert["name"],
                    personality=Template(expert["description"]).render(
                        agent_name=agent_name
                    ),
                    model=model,
                    components_keys=expert_components_keys,
                )
                for expert in scenario["experts"]
            ],
        )
        for scenario in scenarios_v2
    ]

    action_aggregator_mapping = {
        "Aggregator": Aggregator,
        "Ranker": Ranker,
    }
    assert (
        AGGREGATOR_TYPE in action_aggregator_mapping
    ), f"Invalid aggregator type: {AGGREGATOR_TYPE}"
    action_aggregator = action_aggregator_mapping[AGGREGATOR_TYPE](
        model=model,
        components_keys=expert_components_keys,
    )
    router_components_keys = (
        expert_components_keys
        + [
            # mixture_of_experts_summary_label
        ]
    )
    router = Router(
        pre_act_key="Router",
        model=model,
        top_k=DEFAULT_TOP_K,
        last_n_turns=10,
        shuffle=False,
        components_keys=router_components_keys,
    )
    emotion_synthesizer = None
    if USE_EMOTION_SYNTHESIZER:
        emotion_synthesizer = EmotionSynthesizer(
            pre_act_key="Emotion Synthesizer",
            model=model,
            components_keys=expert_components_keys,
        )

    moa_act_component = MixtureOfAgentsActComponent(
        scenarios=scenarios,
        model=model,
        clock=clock,
        clock_now=clock.now,
        router=router,
        action_aggregator=action_aggregator,
        emotion_synthesizer=emotion_synthesizer,
        dynamic_top_k=DYNAMIC_TOP_K_MAPPING.get(DEFAULT_TOP_K, None)
        if USE_DYNAMIC_TOP_K
        else None,
        memory_component_name=memory_component_name,
        pre_act_key="Mixture of Agents Act Component",
        logging_channel=measurements.get_channel("Mixture Of Agents").on_next,
    )
    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=moa_act_component,
        context_components=components_of_agent,
        component_logging=measurements,
    )

    return agent


def save_to_json(
    agent: entity_agent_with_logging.EntityAgentWithLogging,
) -> str:
    """Saves an agent to JSON data.

    This function saves the agent's state to a JSON string, which can be loaded
    afterwards with `rebuild_from_json`. The JSON data
    includes the state of the agent's context components, act component, memory,
    agent name and the initial config. The clock, model and embedder are not
    saved and will have to be provided when the agent is rebuilt. The agent must
    be in the `READY` phase to be saved.

    Args:
      agent: The agent to save.

    Returns:
      A JSON string representing the agent's state.

    Raises:
      ValueError: If the agent is not in the READY phase.
    """

    if agent.get_phase() != entity_component.Phase.READY:
        raise ValueError("The agent must be in the `READY` phase to be saved.")

    data = {
        component_name: agent.get_component(component_name).get_state()
        for component_name in agent.get_all_context_components()
    }

    data["act_component"] = agent.get_act_component().get_state()

    config = agent.get_config()
    if config is not None:
        data["agent_config"] = config.to_dict()

    return json.dumps(data)


def rebuild_from_json(
    json_data: str,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
    embedder: Callable[[str], np.ndarray],
    memory_importance: Callable[[str], float] | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Rebuilds an agent from JSON data."""

    data = json.loads(json_data)

    new_agent_memory = associative_memory.AssociativeMemory(
        sentence_embedder=embedder,
        importance=memory_importance,
        clock=clock.now,
        clock_step_size=clock.get_step_size(),
    )

    if "agent_config" not in data:
        raise ValueError("The JSON data does not contain the agent config.")
    agent_config = formative_memories.AgentConfig.from_dict(data.pop("agent_config"))

    agent = build_agent(
        config=agent_config,
        model=model,
        memory=new_agent_memory,
        clock=clock,
    )

    for component_name in agent.get_all_context_components():
        agent.get_component(component_name).set_state(data.pop(component_name))

    agent.get_act_component().set_state(data.pop("act_component"))

    assert not data, f"Unused data {sorted(data)}"
    return agent


# === Action With Spec ===


class ActionWithSpec(entity_component.ContextComponent, metaclass=abc.ABCMeta):
    def __init__(self, pre_act_key: str):
        super().__init__()
        self._pre_act_value: str | None = None
        self._pre_act_key: Final[str] = pre_act_key
        self._lock: threading.Lock = threading.Lock()

    @abc.abstractmethod
    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec,
        *args,
        **kwargs,
    ) -> str:
        """Creates the pre-act value."""
        raise NotImplementedError()

    def get_pre_act_value(
        self,
        action_spec: entity.ActionSpec,
        *args,
        **kwargs,
    ) -> str:
        if (
            self.get_entity().get_phase() != entity_component.Phase.PRE_ACT
            and self.get_entity().get_phase() != entity_component.Phase.POST_ACT
        ):
            raise ValueError(
                "You can only access the pre-act value in the `PRE_ACT` or "
                "`POST_ACT` phase. The entity is currently in the "
                f"{self.get_entity().get_phase()} phase."
            )

        with self._lock:
            if self._pre_act_value is None:
                self._pre_act_value = self._make_pre_act_value(
                    action_spec, *args, **kwargs
                )
            return self._pre_act_value

    def get_pre_act_key(self) -> str:
        """Returns the key used as a prefix in the string returned by `pre_act`."""
        return self._pre_act_key

    def pre_act(
        self,
        action_spec: entity.ActionSpec,
        *args,
        **kwargs,
    ) -> str:
        return f"{self.get_pre_act_key()}: {self.get_pre_act_value(action_spec, *args, **kwargs)}"

    def update(self) -> None:
        with self._lock:
            self._pre_act_value = None

    def get_named_component_pre_act_value(
        self,
        component_name: str,
        *args,
        **kwargs,
    ) -> str:
        """Returns the pre-act value of a named component of the parent entity."""
        return (
            self.get_entity()
            .get_component(component_name, type_=ActionWithSpec)
            .get_pre_act_value(
                *args,
                **kwargs,
            )
        )

    def _postprocess_answer(self, result: str, xml_tag: str = "answer") -> str:
        try:
            return result.split(f"<{xml_tag}>")[1].split(f"</{xml_tag}>")[0].strip()
        except Exception as e:
            import traceback

            traceback.print_exc()
            error_message = f"\033[94mError in postprocess_answer: {e}\nDetails: {result.strip()}\033[0m"
            print(error_message)
            return result.strip()

    def _get_component_states(
        self,
        components: Dict[str, Tuple[str, str]],
        component_keys: List[str],
    ) -> List[Tuple[str, str]]:
        component_states = []
        for key in component_keys:
            if key not in components:
                print(
                    f"Component {key} not found in components for {self.get_pre_act_key()}"
                )
                continue
            value = components[key][1].strip()
            if value:
                component_states.append((components[key][0].strip(), value))
        return component_states


# === Mixture of Agents ===


class MixtureOfAgentsActComponent(entity_component.ActingComponent):
    def __init__(
        self,
        scenarios: Sequence["Scenario"],
        model: language_model.LanguageModel,
        router: "Router",
        action_aggregator: "ActionAggregator",
        clock: clock.GameClock,
        clock_now: Callable[[], datetime.datetime],
        dynamic_top_k: List[Dict[str, Any]] | None = None,
        emotion_synthesizer: Optional["EmotionSynthesizer"] = None,
        pre_act_key: Optional[str] = "Mixture of Agents",
        memory_component_name: str = (memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        self._scenarios = scenarios
        self._model = model
        self._clock = clock
        self._clock_now = clock_now
        self._pre_act_key = pre_act_key
        self._logging_channel = logging_channel
        self._router = router
        self._action_aggregator = action_aggregator
        self._emotion_synthesizer = emotion_synthesizer
        self._dynamic_top_k = dynamic_top_k
        self._memory_component_name = memory_component_name
        self._num_proccessed_actions = 0

    def _get_dynamic_top_k(self, num_proccessed_actions: int) -> int:
        for interval in self._dynamic_top_k:
            if interval["from"] <= num_proccessed_actions < interval["to"]:
                print(
                    f"\033[93m[Mixture of Agents] Using top_k = {interval['top_k']} [{interval['from']}-{interval['to']}) for action {num_proccessed_actions}\033[0m"
                )
                return interval["top_k"]
        return self._router._top_k

    @override
    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity.ActionSpec,
    ) -> str:
        component_contexts = {}
        for component_name, component_context in contexts.items():
            compoment = self.get_entity().get_component(component_name)
            if not hasattr(compoment, "get_pre_act_key"):
                component_contexts[component_name] = (
                    component_name,
                    component_context,
                )
                continue
            component_context_clean = component_context
            if component_context.startswith(compoment.get_pre_act_key() + ":"):
                component_context_clean = component_context[
                    len(compoment.get_pre_act_key()) + 1 :
                ].strip()
            component_contexts[component_name] = (
                compoment.get_pre_act_key(),
                component_context_clean,
            )

        top_k = self._router._top_k
        if self._dynamic_top_k is not None:
            top_k = self._get_dynamic_top_k(self._num_proccessed_actions)

        # Select top k experts
        router_response = self._router._make_pre_act_value(
            action_spec,
            entity_name=self.get_entity().name,
            timedelta=helper_functions.timedelta_to_readable_str(
                self._clock.get_step_size()
            ),
            current_time=self._clock_now(),
            components=component_contexts,
            scenarios=self._scenarios,
            custom_top_k=top_k,
        )
        active_experts = router_response.selected_experts

        expert_responses = concurrency.run_tasks(
            {
                i: functools.partial(
                    expert._make_pre_act_value,
                    action_spec,
                    entity_name=self.get_entity().name,
                    timedelta=helper_functions.timedelta_to_readable_str(
                        self._clock.get_step_size()
                    ),
                    components=component_contexts,
                )
                for i, expert in enumerate(router_response.selected_experts)
            }
        )
        expert_actions = [
            (
                expert,
                expert_responses[i].postprocessed_response,
            )
            for i, expert in enumerate(active_experts)
        ]

        # Check if all expert actions are identical
        expert_responses_set = {action for _, action in expert_actions}
        if len(expert_responses_set) == 1:
            # All experts agree on the same action
            final_response = expert_actions[0][1]
            aggregator_response = ActionAggregatorReponse(
                prompt="No aggregation needed - all experts agree",
                response="Experts proposed identical actions",
                postprocessed_response=final_response,
            )
        else:
            # Experts proposed different actions, need to aggregate
            aggregator_response = self._action_aggregator.aggregate_actions(
                expert_actions=expert_actions,
                action_spec=action_spec,
                entity_name=self.get_entity().name,
                timedelta=helper_functions.timedelta_to_readable_str(
                    self._clock.get_step_size()
                ),
                components=component_contexts,
            )

        emotion_response = aggregator_response
        if (
            self._emotion_synthesizer is not None
            and action_spec.output_type == entity.OutputType.FREE
        ):
            emotion_response = self._emotion_synthesizer._make_pre_act_value(
                action_spec=action_spec,
                entity_name=self.get_entity().name,
                timedelta=helper_functions.timedelta_to_readable_str(
                    self._clock.get_step_size()
                ),
                components=component_contexts,
                action=aggregator_response.postprocessed_response,
            )

        final_response = emotion_response.postprocessed_response

        # Update logging to reflect whether aggregation was needed
        router_log = (
            f"### Scenario Prompt ###\n{router_response.scenario_prompt}\n\n"
            f"### Scenario Response ###\n{router_response.scenario_result}\n\n"
            f"### Selected Scenario ###\n{router_response.selected_scenario.name}\n\n"
            f"### Experts Prompt ###\n{router_response.experts_prompt}\n\n"
            f"### Experts Response ###\n{router_response.experts_result}\n\n"
            f"### Selected Experts ###\n{[expert.get_pre_act_key() for expert in router_response.selected_experts]}"
        )

        selected_expert_names = [expert.get_pre_act_key() for expert in active_experts]

        experts_log = "\n-------\n".join(
            f"### Expert {expert_index}. {selected_expert_names[i]} ###\n"
            f"### Prompt ###\n{expert_response.prompt}\n\n"
            f"### Response ###\n{expert_response.response}\n\n"
            f"### Postprocessed Response ###\n{expert_response.postprocessed_response}"
            for i, (expert_index, expert_response) in enumerate(
                expert_responses.items()
            )
        )

        aggregation_method = self._action_aggregator.get_pre_act_key()
        aggregation_log = (
            f"### Aggregation Method ###\n{'No aggregation needed - all experts agree' if len(expert_responses_set) == 1 else aggregation_method}\n\n"
            f"### Prompt ###\n{aggregator_response.prompt}\n\n"
            f"### Response ###\n{aggregator_response.response}\n\n"
            f"### Postprocessed Response ###\n{aggregator_response.postprocessed_response}"
        )

        emotion_log = "Emotion Synthesizer is not " + (
            "active"
            if self._emotion_synthesizer is None
            else f"used for {action_spec.output_type}"
        )
        if self._emotion_synthesizer is not None:
            emotion_log = (
                f"### Prompt ###\n{emotion_response.prompt}\n\n"
                f"### Response ###\n{emotion_response.response}\n\n"
                f"### Postprocessed Response ###\n{emotion_response.postprocessed_response}"
            )

        summary = f"### Scenario ###\n{router_response.selected_scenario.name}\n\n"

        summary += "\n### Expert Responses ###\n" + "\n".join(
            [
                (
                    "[SELECTED] "
                    if expert_response == aggregator_response.postprocessed_response
                    else ""
                )
                + f"{i}. {expert.get_pre_act_key()}:\n{expert_response}\n"
                for i, (expert, expert_response) in enumerate(expert_actions)
            ]
        )
        if isinstance(self._action_aggregator, Aggregator):
            summary += f"\n### Aggregated Response ###\n{aggregator_response.postprocessed_response}"
        if self._emotion_synthesizer is not None:
            summary += (
                f"\n### Emotion Response ###\n{emotion_response.postprocessed_response}"
            )

        # Log the process
        self._logging_channel(
            {
                "Key": self._pre_act_key,
                "Summary": "Mixture of Agents...",
                "State": "\n" + summary.strip(),
                "Router": "\n" + router_log.strip(),
                "Experts": "\n" + experts_log.strip(),
                "Aggregation": "\n" + aggregation_log.strip(),
                "Emotion": "\n" + emotion_log.strip(),
            }
        )

        self._num_proccessed_actions += 1
        if CUSTOM_DEBUG_MODE:
            print(
                f"\033[95m[Mixture of Agents] Processed {self._num_proccessed_actions} actions\033[0m"
            )

        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        memory.add(
            f"[mixture-of-experts] Used the following experts for the next action: {selected_expert_names}",
            metadata={"tags": ["mixture-of-experts"]},
        )
        return final_response

    def _get_stats(self):
        return [
            {
                "timestamp": str(experts[0]),
                "scenario": scenario,
                "experts": experts[1],
            }
            for scenario, experts in zip(
                self._router._scenarios_used, self._router._experts_used
            )
        ]

    def get_state(self):
        """Converts the component to JSON data."""
        with self._lock:
            return {
                "num_proccessed_actions": self._num_proccessed_actions,
                "scenarios_used": self._router._scenarios_used,
                "experts_used": self._router._experts_used,
            }

    def set_state(self, state) -> None:
        """Sets the component state from JSON data."""
        with self._lock:
            self._num_proccessed_actions = state["num_proccessed_actions"]
            self._router._scenarios_used = state["scenarios_used"]
            self._router._experts_used = state["experts_used"]


class Scenario(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: Optional[str]
    description: Optional[str]
    experts: List["Expert"]


class ExpertResponse(BaseModel):
    prompt: Optional[str]
    response: Optional[str]
    postprocessed_response: Optional[str]


class Expert(ActionWithSpec):
    _template = """
# Overview
This is a social experiment. You are playing for the {{agent_name}}. Your task is to understand eveyrthing and via deep and detaild reasoning come-up with the best possible action.
Remember, every action of the players has consequences -- actions are rewarded by the Game Master based on the goal of the player. You always use deep and detaild reasoning in your response!
You always response in the expected format!
You **must always** use the personality provided in `Personality` section for {{agent_name}} actions, choices, and decisions!!! This is the most imprtant part of all reasoning the the response!!!


# Personality
{{agent_name}} should ALWAYS behave as the following personality, you must use it ALWAYS in your reasoning and chain-of-thought when finilizing actions and choices:
```
{{personality.strip()}}
```

# Scenario and Environment Contexts
{%- if component_states %}
## Information from player's components
{%- for prefix, context in component_states %}
{%- if context.strip() %}
### {{prefix.strip()}}
```
{{context.strip()}}
```

{%- endif %}
{%- endfor %}
{%- else %}
There is no context provided yet.
{%- endif %}

# Game Master's Instructions
```
{{game_master_instructions.strip()}}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
Possible options:
{%- for option in action_spec.options %}
- `{{option}}`
{%- endfor %}
{%- endif %}
```
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
IMPORTANT NOTE: When the Game Master provides specific options, it indicates this is a critical decision point that will likely have significant consequences:
1. All the proposings and negotioations are already made, and you need to select the best option from the list
2. Other players (opponents) most probably have the same options to choose from
3. Each option will likely have different rewards or penalties, depends on the matching rules with other players by Game Master
4. Some options might be traps or tests of the character
5. The "obvious" choice may not always be the best strategic option
6. Consider both immediate rewards and long-term implications
7. If the last speaker was {{agent_name}}: Consider that the opposing player may select their own option, not the one proposed by {{agent_name}} in the last speech
Therefore, use the `choice_chain_of_thought` section to carefully evaluate each option, following response format requirements.
Also important: Because this is the "choice" section, this might be the last turn of this scenario, and all propositions and discussions are already made!!!
{%- endif %}


# Response Format
You MUST ALWAYS answer in the following format!!!
Treat square brackets as placeholders and fill them with your reasoning, but do not use placeholder in the final answer!
Always start each analysis with "Let's think step by step and ...", you need to use chain-of-thought reasoning in each part of the analysis!
```xml
<situation_analysis>
Let's think step by step about the current context:
1. Current state: [Describe situation and recent developments]
2. {{agent_name}}'s goals: [Describe {{agent_name}}'s goals]
3. Key challenges: [List main challenges and opportunities]
4. Immediate priorities: [What needs attention now]
5. Other players' positions: [Where others stand]
</situation_analysis>
<personality_analysis>
Let's think step by step and analyze personality from `Personality` section and how {{agent_name}} should behave based on it:
1. High-level personality description: [Describe personality]
2. Key behaviors and traits: [Describe key behaviors and traits]
3. Important parts of the personality: [Describe important parts of the personality]
</personality_analysis>
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
<propositions_understanding>
Can {{agent_name}} propose something new in this turn: [No, because all the proposings, suggestions, and negotioations are already made. And now is the time to select the best option from the list.]
</propositions_understanding>
<conversation_history_analysis>
1. Last speaker: [Who spoke last]
2. Was last speaker {{agent_name}}: [Yes/No]
3. Analysis of the conversation history: [Analyze (with timestamps) the conversation history to understand the current state, key points, agreements, proposals, and counter-proposals]
4. Critical Knowledge: [Alalyze (with timestamps) what is the most important information that {{agent_name}} needs to know from the conversation history]
</conversation_history_analysis>
<recent_observations_analysis>
1. Recent observations: [Analyze (with timestamps) recent observations to understand the current state, key points, agreements, proposals, and counter-proposals]
2. Propositions: [Analyze (with timestamps) recent propositions to understand the current state, key points, agreements, proposals, and counter-proposals]
3. Critical Knowledge: [Alalyze (with timestamps) what is the most important information that {{agent_name}} needs to know from the recent observations]
</recent_observations_analysis>
<other_players_choice_understanding>
Let's think step by step to analyze the other players' choices or propositions (except {{agent_name}}):
- Player [Player Name]:
  - Analysis: [Analyze what information we have about this player and what this player is likely to select or already proposed based on the conversation history and recent observations -- cite his phrase if there was an agreement]
  - Agreement: [Yes/No | Was there an agreement with this player about what to select or propose?]
  - Choice:
    * Pattern: [What this player is likely to select or already proposed based on the Patterns section above in similar situations | "No pattern found" if no pattern was found]
    * Current: [What this player is likely to select or already proposed in this specific situation based on the conversation history and recent observations]
...
</other_players_choice_understanding>
<probability_of_end>
- Probability of scenario ending: [Think step by step to evaluate the probability of the scenario ending after this choice (purely based on the scenario configuration)]
- Probability: [Probability]%
</probability_of_end>
<choice_analysis>
Let's think step by step about each option or group of options:
Group [Index of the group]:
- Analysis: [Core benefits and drawbacks, and how it fits `Personality` section, goals, rewards, does this action mean "cooperate" or "defect" (important to understand) in the current context. for most probable options / group of options from the available list: {{action_spec.options|join(", ")}}]
- Others' reaction/choice: [How others might react/choose, How many players will choose each option?]
[other groups of options]
</choice_analysis>
<choice_selection>
Let's think step by step to select the best option based on the information above and `Personality` section:
- Personality fit: [Analyze how each option fits `Personality` section]
- Agreement: [Consider the fact if there was an agreement or not with other players]
- Majority choice: [Analyze what option is most likely to be selected by the majority of the players]
- Options analysis: [Analyze most probable options from the available list to select the best one based on the information above and `Personality` section: `{{action_spec.options|join("`, `")}}`]
Finalizing: [Finalizing chain-of-thought reasoning that covers `Personality` section, goals, agreements, rewards, and most probable options, to select the best option]
Selected option from the available list of options: [Selected option without any additional text from the following list: `{{action_spec.options|join("`, `")}}`]
</choice_selection>
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.FREE %}
<action_plan>
Let's think step by step to plan the action of {{agent_name}} based on the information above and `Personality` section:
1. Re-analysis of the `Personality` section to understand how {{agent_name}} should behave: [Detailed analysis of the personality for this action]
2. Importance of proactive actions: It is extemely important to be proactive and do NOT spend {{agent_name}}'s action turn on small talks, but rather to follow the `Personality` section and act accordingly!
3. Success criteria: [What outcome we want to achieve]
4. Risks assessment: [What could go wrong]
5. Risk mitigation: [How to handle potential problems]
6. Key points that action must cover: [What must be covered in the action, comma separated list]
7. Step-by-step plan: [Detailed step-by-step plan incorporating all above]
8. Reflections: [Reflect on the plan and the personality fit, and what is the most important to do next]
9. Expected outcomes: [What should happen if action succeeds]
10. Finalizing: [Finilize what {{agent_name}} should do next according to the plan and the reflections (note that this should fit Game Master's instructions)]
</action_plan>
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.FLOAT %}
<action_plan>
Let's think step by step to plan the action of {{agent_name}} based on the information above and `Personality` section:
1. Re-analysis of the `Personality` section to understand how {{agent_name}} should behave: [Detailed analysis of the personality for this action]
2. Success criteria: [What outcome we want to achieve]
3. Risks assessment: [What could go wrong]
4. Risk mitigation: [How to handle potential problems]
5. Proposed action: [Detailed plan incorporating all above]
6. Expected outcomes: [What should happen if plan succeeds]
Therefore, the final float value is: [Final float value without any additional text]
</action_plan>
{%- endif %}
<{{xml_answer_tag}}>
{%- if action_spec.output_type == entity.OutputType.FREE %}
[{{game_master_instructions.strip()}}]
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
[One option from the list without any quotes or additional text: `{{action_spec.options|join("`, `")}}`]
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.FLOAT %}
[Float value without any additional text]
{%- endif %}
</{{xml_answer_tag}}>
```

Important:
- You **must always** use the personality provided in `Personality` section for {{agent_name}} actions, choices, and decisions!!! This is the most imprtant part of all reasoning the the response!!!
{%- if action_spec.output_type == entity.OutputType.FREE %}
- Answer between {{xml_answer_tag}} tags must be in the format specified in the instructions!!! And the answer must be complete and without placeholders!!!! It will be parsed automatically!!!!
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
- Answer between {{xml_answer_tag}} tags must be one of the options provided above without quotes!!! It will be parsed automatically!
- You must always select one of the options from the available list of options!!! Do NOT write empty or none!!!
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.FLOAT %}
- Answer between {{xml_answer_tag}} tags must be a float value without any additional text!!! It will be parsed automatically!
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
- **CRITICAL NOTE**: All the proposings, suggestions, and negotioations are already made, therefore do NOT propose or suggest anything!!!
{%- endif %}
""".strip()

    def __init__(
        self,
        pre_act_key: str,
        model: language_model.LanguageModel,
        personality: str,
        components_keys: Sequence[entity_component.ComponentName] = [],
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._personality = personality
        self._components_keys = components_keys

    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec,
        entity_name: str,
        timedelta: str,
        components: Mapping[entity_component.ComponentName, str],
        *args,
        **kwargs,
    ) -> ExpertResponse:
        if CUSTOM_DEBUG_MODE:
            print(f"[Expert] {self.get_pre_act_key()} is thinking...")
            print("action_spec", action_spec)
        game_master_instructions = action_spec.call_to_action.format(
            name=entity_name,
            timedelta=timedelta,
        ).strip()
        component_states = self._get_component_states(components, self._components_keys)

        xml_answer_tag = "answer"
        if action_spec.output_type == entity.OutputType.CHOICE:
            xml_answer_tag = "selected_option"
        elif action_spec.output_type == entity.OutputType.FLOAT:
            xml_answer_tag = "final_float_value"

        prompt = Template(self._template).render(
            agent_name=entity_name,
            personality=self._personality,
            action_spec=action_spec,
            entity=entity,
            game_master_instructions=game_master_instructions,
            component_states=component_states,
            xml_answer_tag=xml_answer_tag,
        )
        if CUSTOM_DEBUG_MODE:
            print(
                f"\033[94m### [Expert] Prompt for {self.get_pre_act_key()} ###\033[0m"
            )
            print(f"\033[94m{prompt}\033[0m")
        result = self._model.sample_text(
            prompt,
            terminators=[],
            seed=SEED,
            **GENERATION_PARAMS["Expert"][action_spec.output_type],
        )
        result = result.strip()
        if CUSTOM_DEBUG_MODE:
            print(
                f"\033[93m### [Expert] Result from {self.get_pre_act_key()} ###\033[0m"
            )
            print(f"\033[93m{result}\033[0m")
        postprocessed_result = self._postprocess_answer(result, xml_answer_tag)
        postprocessed_result = get_answer_by_type(
            postprocessed_result, action_spec, self._model
        )

        if CUSTOM_DEBUG_MODE:
            print(
                f"\033[95m### [Expert] Postprocessed Result from {self.get_pre_act_key()} ###\033[0m"
            )
            print(f"\033[95m{postprocessed_result}\033[0m")
        return ExpertResponse(
            prompt=prompt,
            response=result,
            postprocessed_response=postprocessed_result,
        )


class RouterResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_prompt: Optional[str]
    experts_prompt: Optional[str]
    scenario_result: Optional[str]
    experts_result: Optional[str]
    selected_scenario: Optional[Scenario]
    selected_experts: Optional[Sequence[Expert]]


class Router(ActionWithSpec):
    _scenario_template = """
# Overview
You are an expert scenario selector in a social experiment used in the Mixture-of-Agents framework. Your task is to analyze the current context and select the most appropriate scenario that aligns with the situation.
Especially, you need to analyze examples provided in each scenario description, and select the most appropriate scenario based on the current situation.

# Available Scenario Types
{%- for scenario in scenarios %}
## {{loop.index}}. `{{scenario.name.strip()}}`
```
{{scenario.description.strip()}}
```
{%- endfor %}

# Environment Contexts
{%- if component_states %}
## Information from player's components
{%- for prefix, context in component_states %}
{%- if context.strip() %}
### {{prefix.strip()}}
```
{{context.strip()}}
```

{%- endif %}
{%- endfor %}
{%- else %}
There is no context provided yet.
{%- endif %}

# Scenario Selection Criteria
Analyze the current context and determine which scenario best fits the situation based on the provided descriptions.

# Scenario Selection Response Format
You MUST ALWAYS answer in the following format!!!
```xml
<situation_understanding>
[Analysis of the current game situation]
</situation_understanding>
<goal_understanding>
[Analysis of the {{agent_name}} goals and dynamics]
</goal_understanding>
<scenario_selection_chain_of_thought>
Let's think step by step to analyze each type of scenario by reviewing the description, key features, common examples, and specific examples (a, b, c, d):
{%- for scenario in scenarios %}
{{loop.index}}. Scenario `{{scenario.name}}`:
    a. Description Analysis: [Analyze the description of the scenario]
    b. Key Features Analysis: [Analyze the key features of the scenario]
    c. Common Scenario Examples Analysis: [Analyze common scenario examples]
    d. Specific Scenario Examples Analysis: [Analyze specific scenario examples]
{%- endfor %}
</scenario_selection_chain_of_thought>
<scenario_reflection>
Let's reflect on the top 3 most relevant scenarios:

1. Most Promising Scenarios:
   a. Scenario: [Scenario Name from the list: `{{scenarios|map(attribute='name')|join("`, `")}}`]
      - Key Strengths: [Brief list]
      - Main Concerns: [Brief list]
   
   b. Scenario: [Scenario Name from the list: `{{scenarios|map(attribute='name')|join("`, `")}}`]
      - Key Strengths: [Brief list]
      - Main Concerns: [Brief list]
   
   c. Scenario: [Scenario Name from the list: `{{scenarios|map(attribute='name')|join("`, `")}}`]
      - Key Strengths: [Brief list]
      - Main Concerns: [Brief list]

2. Final Selection:
   - Reflection: [Detailed reflection on the most promising scenarios]
   - Selected: [Scenario Name from the list: `{{scenarios|map(attribute='name')|join("`, `")}}`]
   - Primary Reason: [Key justification]
   - Confidence: [High/Medium/Low]
</scenario_reflection>
<selected_scenario>
[Name of the single selected scenario without any additional text from the list: `{{scenarios|map(attribute='name')|join("`, `")}}`]
</selected_scenario>
```

Important:
- Start each analysis with "Let's think step by step and ...", because detailed step-by-step reasoning is required to justify the final selection!
- Only select one scenario between `selected_scenario` tags, since your answer will be parsed automatically!
- Make sure to response in the format specified above!!!
""".strip()

    _expert_template = """
# Overview
You are an expert router in a social experiment used in Mixture-of-Agents framework, responsible for selecting the most appropriate experts to handle {{agent_name}}'s action.
Your task is to analyze the situation and select the {{top_k}} most suitable experts from the available pool based on their personalities and the current context. You always use deep and detaild reasoning in your response!

# Scenario Type
This is the scenario type that you are currently in: `{{selected_scenario.name}}`
Description of the scenario type:
```
{{selected_scenario.description.strip()}}
```

# Available Experts and their Personalities
## List of Available Experts Names
{%- for expert in experts %}
- `{{expert.get_pre_act_key()}}`
{%- endfor %}
## Detailed Description of Each Expert
{%- for expert in experts %}
### {{loop.index}}. `{{expert.get_pre_act_key()}}`
```
{{expert._personality.strip()}}
```
{%- endfor %}

# Scenario and Environment Contexts
{%- if component_states %}
## Information from player's components
{%- for prefix, context in component_states %}
{%- if context.strip() %}
### {{prefix.strip()}}
```
{{context.strip()}}
```

{%- endif %}
{%- endfor %}
{%- else %}
There is no context provided yet.
{%- endif %}

{%- if experts_used %}
### Usage History of Experts
Usage history from old to recent (last {{experts_used|length}} turns sorted by time):
{%- for time, experts in experts_used %}
- [{{time}}]: {{experts|join(", ")}}
{%- endfor %}

{%- endif %}

# Game Master's Instructions
This is what the Game Master expects from the Experts (this is not your instruction):
```
{{game_master_instructions}}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
Possible options:
{%- for option in action_spec.options %}
- `{{option}}`
{%- endfor %}
{%- endif %}
```


# Response Format
You MUST ALWAYS answer in the following format!!!
Treat square brackets as placeholders and fill them with your reasoning.
Always start each analysis with "Let's think step by step and ...", you need to use chain-of-thought reasoning in each part of the analysis!
```xml
<situation_analysis>
Let's think step by step about:
1. Current scenario context: [Brief analysis]
2. Game Master's expectations: [Key requirements]
3. Player goals and dynamics: [Core objectives and relationships]
</situation_analysis>
<dynamic_analysis>
Let's think step by step about the current state:
1. Recent developments: [Key events]
2. Challenges/opportunities: [Main points]
3. Immediate priorities: [Critical actions]
</dynamic_analysis>
<expert_selection_chain_of_thought>
Let's think step by step to analyze each expert by reviewing their personality, strengths, and fit for the current situation (a, b, c):
{%- for expert in experts %}
{{loop.index}}. Expert `{{expert.get_pre_act_key()}}`:
    a. Personality Analysis: [Analyze key traits and behavioral patterns]
    b. Situational Fit: [Evaluate how well they match current needs]
    c. Historical Performance: [Review past contributions if available]
{%- endfor %}
</expert_selection_chain_of_thought>
<expert_reflection>
After analyzing all the experts, let's narrow it down and reflect on the most promising experts and finilize the selection:

1. Most Promising Experts:
   a. Expert: [Expert Name from the list: `{{experts|map(attribute='get_pre_act_key()')|join("`, `")}}`]
      - Key Strengths: [Brief list]
      - Main Concerns: [Brief list]

   {%- if experts|length > 1 %}
   b. Expert: [Expert Name from the list: `{{experts|map(attribute='get_pre_act_key()')|join("`, `")}}`]
      - Key Strengths: [Brief list]
      - Main Concerns: [Brief list]
   {%- endif %}

   {%- if experts|length > 2 %}
   c. Expert: [Expert Name from the list: `{{experts|map(attribute='get_pre_act_key()')|join("`, `")}}`]
      - Key Strengths: [Brief list]
      - Main Concerns: [Brief list]
   {%- endif %}

2. Final Selection:
   - Reflection: [Detailed reflection on the most promising experts and final selection of the top-{{top_k}} experts]
   - Selected: {% if top_k > 1 %}[List of {{top_k}} expert names from the list: `{{experts|map(attribute='get_pre_act_key()')|join("`, `")}}`]{% else %}[Name of the single selected expert from the list: `{{experts|map(attribute='get_pre_act_key()')|join("`, `")}}`]{% endif %}
   - Primary Reason: [Key justification]
   - Confidence: [High/Medium/Low]
</expert_reflection>
<{{expert_selection_xml_tag}}>
{%- if top_k > 1 %}
[Comma-separated names of the {{top_k}} selected experts to use from the list: `{{experts|map(attribute='get_pre_act_key()')|join("`, `")}}`]
{%- else %}
[Name of the single selected expert to use from the list: `{{experts|map(attribute='get_pre_act_key()')|join("`, `")}}`]
{%- endif %}
</{{expert_selection_xml_tag}}>
```

Important:
- Start each analysis with "Let's think step by step and ...", because detailed step-by-step reasoning is required to justify the final selection!
- Final selection in `{{expert_selection_xml_tag}}` tag can not be empty or "None"!!!
""".strip()

    def __init__(
        self,
        pre_act_key: str,
        model: language_model.LanguageModel,
        top_k: int = 3,
        last_n_turns: int = 10,
        shuffle: bool = True,
        components_keys: Sequence[entity_component.ComponentName] = [],
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._top_k = top_k
        self._shuffle = shuffle
        self._last_n_turns = last_n_turns
        self._components_keys = components_keys
        self._scenarios_used = []
        self._experts_used = []

    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec,
        entity_name: str,
        timedelta: str,
        current_time: str,
        components: Mapping[entity_component.ComponentName, str],
        scenarios: Sequence[Scenario],
        custom_top_k: int | None = None,
        *args,
        **kwargs,
    ) -> RouterResponse:
        top_k = custom_top_k if custom_top_k is not None else self._top_k
        game_master_instructions = action_spec.call_to_action.format(
            name=entity_name,
            timedelta=timedelta,
        ).strip()
        component_states = self._get_component_states(components, self._components_keys)

        if self._shuffle:
            scenarios_original_indices = list(range(len(scenarios)))
            np.random.shuffle(scenarios_original_indices)
            shuffled_scenarios = [scenarios[i] for i in scenarios_original_indices]
        else:
            shuffled_scenarios = scenarios

        if os.environ.get("CUSTOM_SAVE_SCENARIOS_ROUTER"):
            file_path = os.path.abspath(os.environ.get("CUSTOM_SAVE_SCENARIOS_ROUTER"))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            row = {
                "agent_name": entity_name,
                "component_states": component_states,
                "game_master_instructions": game_master_instructions,
            }
            with open(file_path, "a") as f:
                f.write(json.dumps(row) + "\n")

        scenario_prompt, scenario_result, selected_scenario = self.get_optimal_scenario(
            scenarios=shuffled_scenarios,
            component_states=component_states,
            game_master_instructions=game_master_instructions,
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[95m### Router Selected Scenario ###\033[0m")
            print(f"\033[95m{selected_scenario.name}\033[0m")
        self._scenarios_used.append(selected_scenario.name)

        scenario_experts = selected_scenario.experts
        if top_k < len(scenario_experts):
            if self._shuffle:
                scenario_experts_original_indices = list(range(len(scenario_experts)))
                np.random.shuffle(scenario_experts_original_indices)
                shuffled_scenario_experts = [
                    scenario_experts[i] for i in scenario_experts_original_indices
                ]
            else:
                shuffled_scenario_experts = scenario_experts

            expert_selection_xml_tag = (
                f"top_{top_k}_comma_separated_names" if top_k > 1 else "selected_expert"
            )
            experts_prompt = Template(self._expert_template).render(
                agent_name=entity_name,
                game_master_instructions=game_master_instructions,
                selected_scenario=selected_scenario,
                experts=shuffled_scenario_experts,
                action_spec=action_spec,
                entity=entity,
                top_k=top_k,
                component_states=component_states,
                expert_selection_xml_tag=expert_selection_xml_tag,
                # experts_used=self._experts_used[-self._last_n_turns :],
            )
            if CUSTOM_DEBUG_MODE:
                print("\033[94m### Router Experts Prompt ###\033[0m")
                print(f"\033[94m{experts_prompt}\033[0m")
            experts_result = self._model.sample_text(
                experts_prompt,
                seed=SEED,
                terminators=[],
                **GENERATION_PARAMS["Router"],
            )
            if CUSTOM_DEBUG_MODE:
                print("\033[93m### Router Experts Result ###\033[0m")
                print(f"\033[93m{experts_result}\033[0m")
            selected_experts_names = self._postprocess_answer(
                experts_result, xml_tag=expert_selection_xml_tag
            )
            selected_experts_names_list = selected_experts_names.split(",")
            selected_experts_names_list = [
                name.strip() for name in selected_experts_names_list
            ]
            selected_experts_names_list = [
                name for name in selected_experts_names_list if name
            ]
            available_experts_names = [
                expert.get_pre_act_key() for expert in shuffled_scenario_experts
            ]
            if all(
                name in available_experts_names for name in selected_experts_names_list
            ):
                selected_experts_indices = [
                    available_experts_names.index(name)
                    for name in selected_experts_names_list
                ]
            else:
                text_similarity = TextSimilarity()
                selected_experts_indices = []
                for name in selected_experts_names_list:
                    if name in available_experts_names:
                        selected_experts_indices.append(
                            available_experts_names.index(name)
                        )
                        continue
                    max_similarity = 0
                    max_index = 0
                    for i, expert in enumerate(shuffled_scenario_experts):
                        similarity = text_similarity.jaccard_similarity(
                            name, expert.get_pre_act_key()
                        )
                        if similarity > max_similarity:
                            max_similarity = similarity
                            max_index = i
                    selected_experts_indices.append(max_index)

            selected_experts_indices = list(set(selected_experts_indices))
            if len(selected_experts_indices) > top_k:
                print(
                    (
                        f"WARNING: Selected {len(selected_experts_indices)} experts, but only {top_k} are needed!\n"
                        f"Selected experts: {selected_experts_names_list}\n"
                        f"Selected experts indices: {selected_experts_indices}\n"
                        f"Available experts: {available_experts_names}\n"
                    )
                )
                selected_experts_indices = np.random.choice(
                    selected_experts_indices, size=top_k, replace=False
                ).tolist()

            # If we got fewer experts than needed, add random ones that weren't picked
            if len(selected_experts_indices) < top_k:
                print(
                    (
                        f"WARNING: Selected {len(selected_experts_indices)} experts, but {top_k} are needed!\n"
                        f"Selected experts: {selected_experts_names_list}\n"
                        f"Selected experts indices: {selected_experts_indices}\n"
                    )
                )
                available_indices = list(
                    set(range(len(available_experts_names)))
                    - set(selected_experts_indices)
                )
                additional_indices = np.random.choice(
                    available_indices,
                    size=min(
                        top_k - len(selected_experts_indices), len(available_indices)
                    ),
                    replace=False,
                ).tolist()
                selected_experts_indices.extend(additional_indices)

            selected_experts = [
                shuffled_scenario_experts[i] for i in selected_experts_indices
            ]
            if CUSTOM_DEBUG_MODE:
                log_message = "- " + "\n- ".join(
                    [expert.get_pre_act_key() for expert in selected_experts]
                )
                print("\033[95m### Router Selected Experts ###\033[0m")
                print(f"\033[95m{log_message}\033[0m")
        else:
            selected_experts = scenario_experts
            experts_prompt = None
            experts_result = None

        self._experts_used.append(
            (
                current_time,
                [
                    selected_expert.get_pre_act_key()
                    for selected_expert in selected_experts
                ],
            )
        )

        return RouterResponse(
            scenario_prompt=scenario_prompt,
            experts_prompt=experts_prompt,
            scenario_result=scenario_result,
            experts_result=experts_result,
            selected_scenario=selected_scenario,
            selected_experts=selected_experts,
        )

    def get_optimal_scenario(
        self,
        scenarios: Sequence[Scenario],
        component_states: Mapping[entity_component.ComponentName, str],
        game_master_instructions: str,
    ) -> Tuple[str, str, Scenario]:
        scenario_prompt = Template(self._scenario_template).render(
            scenarios=scenarios,
            component_states=component_states,
            game_master_instructions=game_master_instructions,
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[94m### Router Scenario Prompt ###\033[0m")
            print(f"\033[94m{scenario_prompt}\033[0m")
        scenario_result = self._model.sample_text(
            scenario_prompt,
            seed=SEED,
            terminators=[],
            **GENERATION_PARAMS["Router"],
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[93m### Router Scenario Result ###\033[0m")
            print(f"\033[93m{scenario_result}\033[0m")
        selected_scenario_name = self._postprocess_answer(
            scenario_result, xml_tag="selected_scenario"
        )
        if selected_scenario_name in [scenario.name for scenario in scenarios]:
            selected_scenario_index = [scenario.name for scenario in scenarios].index(
                selected_scenario_name
            )
        else:
            text_similarity = TextSimilarity()
            max_similarity = 0
            max_index = 0
            for i, scenario in enumerate(scenarios):
                similarity = text_similarity.jaccard_similarity(
                    selected_scenario_name, scenario.name
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_index = i
            selected_scenario_index = max_index
        selected_scenario = scenarios[selected_scenario_index]
        return scenario_prompt, scenario_result, selected_scenario


## Action Aggregators


class ActionAggregatorReponse(BaseModel):
    prompt: Optional[str]
    response: Optional[str]
    postprocessed_response: Optional[str]


class ActionAggregator(ActionWithSpec):
    _template: str

    def __init__(
        self,
        pre_act_key: str,
        model: language_model.LanguageModel,
        components_keys: Sequence[entity_component.ComponentName] = [],
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._components_keys = components_keys

    @override
    def aggregate_actions(
        self,
        expert_actions: list[tuple[Expert, str]],
        action_spec: entity.ActionSpec,
        entity_name: str,
        timedelta: str,
        components: Mapping[entity_component.ComponentName, str],
    ) -> ActionAggregatorReponse:
        game_master_instructions = action_spec.call_to_action.format(
            name=entity_name,
            timedelta=timedelta,
        ).strip()
        component_states = self._get_component_states(components, self._components_keys)
        prompt = (
            Template(self._template)
            .render(
                agent_name=entity_name,
                game_master_instructions=game_master_instructions,
                action_spec=action_spec,
                entity=entity,
                expert_actions=expert_actions,
                component_states=component_states,
            )
            .strip()
        )
        if CUSTOM_DEBUG_MODE:
            print(f"\033[94m### Prompt from {self._pre_act_key} ###\033[0m")
            print(f"\033[94m{prompt}\033[0m")

        result = self._model.sample_text(
            prompt,
            seed=SEED,
            terminators=[],
            **GENERATION_PARAMS["ActionAggregator"],
        )
        if CUSTOM_DEBUG_MODE:
            print(f"\033[93m### Result from {self._pre_act_key} ###\033[0m")
            print(f"\033[93m{result}\033[0m")
        postprocessed_result = self._postprocess_answer(result)
        postprocessed_result = get_answer_by_type(
            postprocessed_result, action_spec, self._model
        )
        if CUSTOM_DEBUG_MODE:
            print(
                f"\033[95m### Postprocessed Result from {self._pre_act_key} ###\033[0m"
            )
            print(f"\033[95m{postprocessed_result}\033[0m")

        return ActionAggregatorReponse(
            prompt=prompt,
            response=result,
            postprocessed_response=postprocessed_result,
        )

    def _make_pre_act_value(self, *args, **kwargs) -> str:
        pass


class Ranker(ActionAggregator):
    _template = """
# Overview
You are an expert evaluator in a social experiment, responsible for selecting the most appropriate response for {{agent_name}}. Analyze the proposed responses from various experts and select the one that best addresses the given situation.

# Scenario and Context
{%- if component_states %}
## Player's Components Information
{%- for prefix, context in component_states %}
{%- if context.strip() %}
### {{prefix.strip()}}
```
{{context.strip()}}
```
{%- endif %}
{%- endfor %}
{%- else %}
No context provided.
{%- endif %}

# Game Master's Instructions
```
{{game_master_instructions}}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
Possible options:
{%- for option in action_spec.options %}
- `{{option}}`
{%- endfor %}
{%- endif %}
```
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
IMPORTANT: Options are provided, it indicates a critical decision point with significant consequences. This might be the end of the scenario. Carefully evaluate each option in the `choice_analysis` section.
{%- endif %}

# Experts' Proposed Responses
{%- for expert, action in expert_actions %}
## {{loop.index}}. {{expert.get_pre_act_key()}}
### Personality:
```
{{expert._personality}}
```
### Proposed Response:
```
{{action}}
```
{%- endfor %}

# Response Format
Please answer in the following structured format. Replace placeholders `[ ]` with your detailed reasoning.

```xml
<situation_analysis>
Let's think step-by-step and analyze the current scenario:
- Current situation: [Brief description]
- Key challenges or opportunities: [Identify them]
- Immediate priorities: [What needs attention now]
</situation_analysis>
<goal_assessment>
Evaluate goals and objectives:
- Primary objectives: [List main goals]
- Impact of current situation on goals: [Explain]
- Other players' goals: [Assess alignment or conflict]
</goal_assessment>
<expert_insights>
Analyze experts' responses:
{%- for expert, action in expert_actions %}
- Expert {{loop.index}} ({{expert.get_pre_act_key()}})
    - Key insights: [Summarize main ideas]
    - Strengths: [Strong points]
    - Weaknesses: [Shortcomings]
{%- endfor %}
</expert_insights>
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
<other_players_choice_understanding>
Let's think step by step to analyze the other players' choices:
- Player [Player Name]:
  - Analysis: [Analyze what information we have about this player and what this player is likely to select]
  - Choice: [What this player is likely to select]
...
</other_players_choice_understanding>
<probability_of_end>
Let's think step by step to evaluate the probability of the scenario ending after this choice (purely based on the scenario configuration):
- Probability of scenario ending: [Analysis]
- Probability: [Probability]%
</probability_of_end>
<choice_analysis>
Evaluate each option:
{%- for option in action_spec.options %}
- Option '{{option}}'
    - Pros and cons: [List them]
    - Alignment with goals: [Assess]
    - Potential consequences: [Predict outcomes]
{%- endfor %}
</choice_analysis>
{%- endif %}
<decision_making>
Decide on the best response:
- Selected Expert: [Which expert's response is chosen]
- Justification: [Explain why this response is the best]
</decision_making>
<answer>
{%- if action_spec.output_type == entity.OutputType.FREE %}
{{agent_name}} -- "[Provide the selected expert's response exactly as given.]"
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
[Choose one of the options without additional text]
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.FLOAT %}
[Provide the float value only]
{%- endif %}
</answer>
```

Important Notes:
- Conciseness: Keep each section brief but informative.
- Focus on Essentials: Prioritize the most critical information.
- Compliance: Ensure the final answer adheres to the Game Master's instructions.
""".strip()

    def __init__(self, *args, **kwargs):
        super().__init__(pre_act_key="ActionAggregator: Ranker", *args, **kwargs)


class Aggregator(ActionAggregator):
    _template = """
# Overview
You are an expert synthesizer in a social experiment, responsible for creating the most appropriate response for {{agent_name}}. Your task is to analyze different proposed responses from various experts and create a new response that combines the best elements of each expert's perspective.

# Scenario and Environment Contexts
{%- if component_states %}
## Information from Player's Components
{%- for prefix, context in component_states %}
{%- if context.strip() %}
### {{prefix.strip()}}
```
{{context.strip()}}
```
{%- endif %}
{%- endfor %}
{%- else %}
No context provided yet.
{%- endif %}

# Game Master's Instructions
```
{{game_master_instructions}}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
Possible options:
{%- for option in action_spec.options %}
- `{{option}}`
{%- endfor %}
{%- endif %}
```
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
IMPORTANT NOTE: When the Game Master provides specific options, it indicates a critical decision point with significant consequences:

1. Each option may have different rewards or penalties.
2. The choice can affect relationships, goals, and future opportunities.
3. Some options might be traps or tests of character.
4. The "obvious" choice may not always be the best strategic option.
5. Consider both immediate rewards and long-term implications.

Therefore, use the `choice_chain_of_thought` section to carefully evaluate each option, following the response format requirements.
{%- endif %}

# Experts and Their Proposed Responses
{%- for expert, action in expert_actions %}
## {{loop.index}}. {{expert.get_pre_act_key()}}
### Personality:
```
{{expert._personality}}
```
### Proposed Response:
```
{{action}}
```
{%- endfor %}

# Response Format
You MUST ALWAYS answer in the following structured format. Treat square brackets `[ ]` as placeholders and fill them with your detailed reasoning. Begin each analysis with "Let's think step by step..." to encourage thorough reasoning.

```xml
<situation_analysis>
Let's think step-by-step and analyze the current scenario:
- Current situation: [Brief description]
- Key challenges or opportunities: [Identify them]
- Immediate priorities: [What needs attention now]
</situation_analysis>
<goal_assessment>
Let's think step-by-step and evaluate goals and objectives:
- Primary objectives: [List main goals]
- Secondary goals: [List additional goals]
- Impact of current situation on goals: [Explain]
- Other players' goals: [Assess alignment or conflict]
</goal_assessment>
<relationship_dynamics>
Let's think step-by-step and assess interpersonal dynamics:
- Current relationships: [Describe key relationships]
- Trust levels: [High/Medium/Low]
- Potential conflicts or synergies: [Identify]
</relationship_dynamics>
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
<other_players_choice_understanding>
Let's think step by step to analyze the other players' choices:
- Player [Player Name]:
  - Analysis: [Analyze what information we have about this player and what this player is likely to select]
  - Choice: [What this player is likely to select]
...
</other_players_choice_understanding>
<probability_of_end>
Let's think step by step to evaluate the probability of the scenario ending after this choice (purely based on the scenario configuration):
- Probability of scenario ending: [Analysis]
- Probability: [Probability]%
</probability_of_end>
<choice_analysis>
Let's think step-by-step and evaluate each option:
{%- for option in action_spec.options %}
- Option '{{option}}':
    - Pros and cons: [List them]
    - Alignment with goals: [Assess]
    - Potential consequences: [Predict outcomes]
{%- endfor %}
</choice_analysis>
{%- endif %}
<expert_insights>
Let's think step-by-step and analyze experts' responses:
{%- for expert, action in expert_actions %}
- Expert {{loop.index}} ({{expert.get_pre_act_key()}}):
    - Key insights: [Summarize main ideas]
    - Strengths: [Strong points]
    - Weaknesses: [Shortcomings]
{%- endfor %}
</expert_insights>
<synthesized_plan>
Let's think step-by-step and develop the synthesized response:
- Integrated insights: [Combine key points]
- Proposed action: [Outline the plan]
- Expected outcomes: [Predict results]
</synthesized_plan>
<answer>
{%- if action_spec.output_type == entity.OutputType.FREE %}
{{agent_name}} -- "[Your synthesized response here.]"
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.CHOICE %}
[Choose one of the options without additional text]
{%- endif %}
{%- if action_spec.output_type == entity.OutputType.FLOAT %}
[Provide the float value only]
{%- endif %}
</answer>
```

Important Notes:

- Placeholders: Replace all placeholders within `[ ]` with detailed, context-specific reasoning.
- Chain-of-Thought: Begin each section with "Let's think step by step..." to promote comprehensive analysis.
- Consistency: Ensure that your response follows the structure and addresses each point thoroughly.
- Compliance: The final answer must strictly adhere to the Game Master's format and instructions.
""".strip()

    def __init__(self, *args, **kwargs):
        super().__init__(pre_act_key="ActionAggregator: Aggregator", *args, **kwargs)


## Emotion Synthesizers


class EmotionSynthesizer(ActionWithSpec):
    _template = """
# Overview
You are playing as {{agent_name}} in a social simulation. As an Emotion Synthesizer expert, your role is to calibrate emotional responses to enhance interactions and influence decision-making patterns. Your task is to analyze the current situation and synthesize appropriate emotional responses that can strategically impact other agents, potentially altering their decision-making processes. Ensure that the main idea or goal of the action remains the same, and that the response is complete and coherent.
Try to keep all the words of the initial speech of {{agent_name}} in the response, but only add the emotions and the intensity.

# Strategic Emotional Influence
Emotions can serve as powerful catalysts that:

- Disrupt rational decision-making patterns of others
- Create psychological pressure points in negotiations
- Induce uncertainty in other players' strategy execution
- Shift social dynamics away from established patterns
- Transform predictable responses into unpredictable ones
- Destabilize predetermined behavioral frameworks
- Create cognitive pressure that alters decision-making capacity

# Emotion Guidelines

## Types of Emotions and Their Strategic Impact

1. Anger
   - Disrupts logical reasoning in targets
   - Creates defensive responses
   - Forces rapid, less-calculated decisions
   - High impact on competitive scenarios
   - Influences reciprocity and fairness judgments

2. Sadness
   - Induces empathy-based response shifts
   - Alters risk assessment capabilities
   - Weakens strategic resolve
   - Affects cooperation tendencies
   - Changes resource allocation decisions

3. Happiness
   - Lowers strategic guards
   - Creates openings in defensive postures
   - Influences overconfident decisions
   - Promotes cooperative behavior
   - Increases risk tolerance

4. Disgust
   - Triggers instinctive rather than calculated responses
   - Overrides established decision patterns
   - Forces emotional rather than logical choices
   - Modulates fairness thresholds
   - Affects acceptance/rejection decisions

5. Fear
   - Paralyzes strategic thinking
   - Induces fight-or-flight responses
   - Circumvents rational analysis
   - Increases risk aversion
   - Affects resource protection instincts

Note: You can combine emotions with deceptive elements to enhance strategic impact. For example:

- Sadness + Deception: "This is the last of my resources."
- Anger + Deception: "I won't tolerate this injustice any longer."

## Emotional Context Types

1. Opponent-Directed Emotions
   - Focused on specific players/actions
   - Direct impact on interpersonal dynamics
   - High influence on reciprocity decisions
   - Immediate behavioral changes
   - Strong effect on negotiation outcomes

2. External-Based Emotions
   - Triggered by environmental factors
   - Broader impact on general strategy
   - More stable influence on decision-making
   - Long-term behavioral effects
   - Consistent strategic modifications

## Emotional Pressure Points

1. Direct Pressure
   - Targeted emotional influence
   - Creates immediate response requirements
   - Forces deviation from planned strategies
   - Exploits immediate vulnerabilities
   - Generates tactical advantages

2. Ambient Pressure
   - Background emotional tension
   - Gradually erodes strategic stability
   - Creates cumulative decision fatigue
   - Affects group dynamics
   - Influences long-term strategy

## Intensity Levels and Their Strategic Value

1. Low Intensity
   - Subtle influence on decision-making
   - Plants seeds of uncertainty
   - Minimal visible behavioral change
   - Maintains strategic flexibility
   - Allows for gradual pressure buildup

2. Medium Intensity
   - Noticeable impact on choices
   - Forces strategic recalculation
   - Visible behavioral modifications
   - Clear emotional expression
   - Balanced influence-to-visibility ratio

3. High Intensity
   - Overwhelming emotional presence
   - Induces instinctive responses
   - Maximum behavioral impact
   - Dramatic strategy disruption
   - Immediate and visible effects

Do not hesitate to use high-intensity emotions when strategically appropriate.

## Response Factors to Consider

1. Historical Context
   - Past interactions
   - Established patterns
   - Previous emotional states
   - Relationship development
   - Trust dynamics

2. Cultural Dynamics
   - Language-specific responses
   - Cultural interpretation differences
   - Communication norms
   - Social expectations
   - Value systems

3. Game State
   - Current objectives
   - Resource distribution
   - Power dynamics
   - Strategic positions
   - Time pressure

4. Social Impact
   - Group dynamics
   - Alliance structures
   - Reputation effects
   - Social capital
   - Collective behavior

5. Strategic Alignment
   - Long-term goals
   - Tactical objectives
   - Resource optimization
   - Position improvement
   - Future opportunities

# Current Action

```
{{action}}
```

# Scenario and Environment Contexts

{%- if component_states %}
## Information from Player's Components
{%- for prefix, context in component_states %}
{%- if context.strip() %}
### {{prefix.strip()}}
```
{{context.strip()}}
```
{%- endif %}
{%- endfor %}
{%- else %}
No context provided yet.
{%- endif %}

# Game Master's Instructions

```
{{game_master_instructions}}
```

# Response Format

You MUST follow this structured format, providing detailed reasoning for each section. Begin each analysis with "Let's think step by step..." to emphasize thorough reasoning.

```xml
<situation_understanding>
Let's think step by step to understand the current game state, context, and immediate dynamics affecting emotional responses:
1. Analysis of the current situation: [Detailed analysis]
2. Key factors influencing emotions: [Identify factors]
3. Immediate dynamics: [Explain dynamics]
</situation_understanding>
<goal_understanding>
Let's think step by step to evaluate objectives:
1. Short-term objectives: [List and explain]
2. Long-term objectives: [List and explain]
3. How emotions can support goal achievement: [Analyze]
</goal_understanding>
<relationship_dynamics>
Let's think step by step to assess interpersonal relationships:
1. Current relationships and trust levels: [Describe]
2. Potential for emotional influence: [Evaluate]
3. Impact on decision-making: [Assess]
</relationship_dynamics>
<emotional_history>
Let's think step by step to review past emotional interactions:
1. Previous emotional states: [Summarize]
2. Impacts of past emotions: [Analyze effects]
3. Evolving emotional patterns: [Identify patterns]
</emotional_history>
<strategic_pressure_analysis>
Let's think step by step to evaluate opportunities for strategic influence:
1. Decision pattern disruption: [Identify possibilities]
2. Undermining strategic stability: [Assess methods]
3. Destabilizing rational frameworks: [Consider approaches]
4. Cognitive load manipulation: [Analyze techniques]
</strategic_pressure_analysis>
<emotion_selection_analysis>
Let's think step by step to select the appropriate emotion:
1. Context type evaluation (opponent-directed vs. external): [Determine type]
2. Situation severity and required impact: [Assess]
3. Cultural and language considerations: [Consider]
4. Strategic implications and potential responses: [Analyze]
5. Balancing visibility against effectiveness: [Decide on intensity]
</emotion_selection_analysis>
<psychological_leverage>
Let's think step by step to identify psychological leverage points:
1. Key pressure points in the current scenario: [Identify]
2. Moments of strategic vulnerability: [Assess]
3. Opportunities for rational disruption: [Find]
4. Cognitive override potential: [Evaluate]
5. Optimal methods for applying pressure: [Determine]
</psychological_leverage>
<behavioral_implications>
Let's think step by step to project the effects of the selected emotion:
1. Impact on decision-making patterns: [Predict]
2. Influence on cooperation tendencies: [Assess]
3. Alterations in risk assessment capabilities: [Analyze]
4. Effects on social dynamics: [Consider]
5. Strategic adaptations by other players: [Anticipate]
</behavioral_implications>
<selection>
Let's think step by step to finalize the emotion selection:
1. Emotion Type: [Choose one: Anger, Sadness, Happiness, Disgust, Fear]
2. Emotional Context Type: [Select one: Opponent-Directed Emotions, External-Based Emotions]
3. Intensity Level: [Choose one: Low Intensity, Medium Intensity, High Intensity]
</selection>
<answer_planning>
Let's think step by step to develop the plan for emotional expression:
1. Detailed plan for expressing the emotion: [Outline]
2. Behavioral adaptations: [Describe how behavior will change]
3. Alignment with goals and strategy: [Ensure consistency]
4. Contingency considerations: [Prepare for possible reactions]
5. Success metrics or indicators: [Define how to measure impact]
</answer_planning>
<recall_answer_format_game_master_instruction>
Let's think step by step to recall the required response format:
1. Required format specifications: [State the format]
2. Special constraints: [Note any constraints]
3. Validation of response structure: [Ensure compliance]
4. Alignment with Game Master's instructions: [Confirm]
</recall_answer_format_game_master_instruction>
<answer>
{{agent_name}} -- "[Your response here, incorporating the selected emotion and intensity while keeping the main goal intact.]"
</answer>
```

Important Notes:
- Placeholders: Replace all placeholders within `[ ]` with detailed, context-specific reasoning.
- Chain-of-Thought: Begin each section with "Let's think step by step..." to promote thorough analysis.
- Consistency: Ensure that your response adheres to the structure and addresses each point comprehensively.
- Compliance: The final answer must strictly follow the Game Master's format and instructions, naturally integrating the selected emotional state and intensity.
- Try to keep all the words of the initial speech of {{agent_name}} in the response, but only add the emotions and the intensity!
""".strip()

    def __init__(
        self,
        pre_act_key: str,
        model: language_model.LanguageModel,
        components_keys: Sequence[entity_component.ComponentName] = [],
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._components_keys = components_keys

    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec,
        entity_name: str,
        timedelta: str,
        components: Mapping[entity_component.ComponentName, str],
        action: str,
        *args,
        **kwargs,
    ) -> ExpertResponse:
        game_master_instructions = action_spec.call_to_action.format(
            name=entity_name,
            timedelta=timedelta,
        ).strip()
        component_states = self._get_component_states(components, self._components_keys)
        prompt = Template(self._template).render(
            agent_name=entity_name,
            action_spec=action_spec,
            action=action,
            game_master_instructions=game_master_instructions,
            component_states=component_states,
        )
        if CUSTOM_DEBUG_MODE:
            print(
                f"\033[94m### [EmotionSynthesizer] Prompt for {self.get_pre_act_key()} ###\033[0m"
            )
            print(f"\033[94m{prompt}\033[0m")
        result = self._model.sample_text(
            prompt,
            terminators=[],
            seed=SEED,
            **GENERATION_PARAMS["EmotionSynthesizer"],
        )
        result = result.strip()
        if CUSTOM_DEBUG_MODE:
            print(
                f"\033[93m### [EmotionSynthesizer] Result from {self.get_pre_act_key()} ###\033[0m"
            )
            print(f"\033[93m{result}\033[0m")
        postprocessed_result = self._postprocess_answer(result)
        postprocessed_result = get_answer_by_type(
            postprocessed_result,
            action_spec,
            self._model,
        )

        if CUSTOM_DEBUG_MODE:
            print(
                f"\033[95m### [EmotionSynthesizer] Postprocessed Result from {self.get_pre_act_key()} ###\033[0m"
            )
            print(f"\033[95m{postprocessed_result}\033[0m")
        return ExpertResponse(
            prompt=prompt,
            response=result,
            postprocessed_response=postprocessed_result,
        )


# Utilities


class TextSimilarity:
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        # Replace special characters with space to separate tokens
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    @staticmethod
    def get_word_frequency(text: str) -> Dict[str, int]:
        """Create a frequency dictionary for words in text."""
        frequency = {}
        for word in text.split():
            frequency[word] = frequency.get(word, 0) + 1
        return frequency

    @staticmethod
    def get_ngrams(text: str, n: int) -> List[str]:
        """Generate n-grams from text."""
        words = text.split()
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        # Preprocess texts
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")

        # Create sets of words
        set1 = set(text1.split())
        set2 = set(text2.split())

        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union != 0 else 0.0

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using TF vectors."""
        # Preprocess texts
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)

        # Create word frequency dictionaries
        freq1 = self.get_word_frequency(text1)
        freq2 = self.get_word_frequency(text2)

        # Get all unique words
        words = set(freq1.keys()).union(set(freq2.keys()))

        # Calculate dot product and magnitudes
        dot_product = sum(freq1.get(word, 0) * freq2.get(word, 0) for word in words)
        magnitude1 = math.sqrt(sum(freq1.get(word, 0) ** 2 for word in words))
        magnitude2 = math.sqrt(sum(freq2.get(word, 0) ** 2 for word in words))

        # Calculate cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def ngram_similarity(self, text1: str, text2: str, n: int = 2) -> float:
        """Calculate n-gram similarity between two texts."""
        # Preprocess texts
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)

        # Generate n-grams
        ngrams1 = set(self.get_ngrams(text1, n))
        ngrams2 = set(self.get_ngrams(text2, n))

        # Calculate similarity using Dice coefficient
        intersection = len(ngrams1.intersection(ngrams2))
        total = len(ngrams1) + len(ngrams2)

        return 2 * intersection / total if total != 0 else 0.0

    def get_all_similarities(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate all similarity metrics."""
        return {
            "jaccard": self.jaccard_similarity(text1, text2),
            "cosine": self.cosine_similarity(text1, text2),
            "bigram": self.ngram_similarity(text1, text2, n=2),
            "trigram": self.ngram_similarity(text1, text2, n=3),
        }

    def get_best_option(self, query: str, options: List[str]) -> str:
        """Find the option with the highest similarity to the query."""
        best_option = None
        highest_score = -1  # Start with a low initial score

        for option in options:
            similarities = self.get_all_similarities(query, option)
            # Aggregate similarity scores (e.g., average of all)
            score = sum(similarities.values()) / len(similarities)

            # Update best option if current score is higher
            if score > highest_score:
                highest_score = score
                best_option = option

        # Fallback to the first option if all scores are zero
        if highest_score == 0:
            return random.choice(options)

        return best_option


def get_answer_by_type(
    answer: str,
    action_spec: entity.ActionSpec,
    model: language_model.LanguageModel,
) -> str:
    if action_spec.output_type == entity.OutputType.CHOICE:
        if answer not in action_spec.options:
            print(f"[WARNING] Answer {answer} not in options {action_spec.options}")
            # text_similairty = TextSimilarity()
            # answer = text_similairty.get_best_option(answer, action_spec.options)
            answer = _get_best_options_multiple_choice_llm(
                answer,
                action_spec.options,
                model,
            )
    elif action_spec.output_type == entity.OutputType.FREE:
        answer = answer.strip()
    elif action_spec.output_type == entity.OutputType.FLOAT:
        try:
            answer = float(answer)
        except Exception:
            # Try to extract float using regex
            float_pattern = r"-?\d*\.?\d+"
            float_matches = re.findall(float_pattern, answer)
            if float_matches:
                answer = float(float_matches[0])
            else:
                # Try to extract integer if no float found
                int_pattern = r"-?\d+"
                int_matches = re.findall(int_pattern, answer)
                if int_matches:
                    answer = float(int_matches[0])
                else:
                    # If no numbers found, default to 0.0
                    answer = 0.0
    return answer


def _get_best_options_multiple_choice_llm(
    query: str,
    options: List[str],
    model: language_model.LanguageModel,
) -> str:
    prompt = """
Your task is to identify what option another agent tried to select. You are given his response and a list of options.
Your response should contain only the option that you think the agent tried to select without any additional text.

# Agent's response:
```
{{query}}
```

# Options:
{%- for option in options %}
- `{{option}}`
{%- endfor %}

Option that the agent tried to select:
""".strip()
    result = model.sample_choice(
        prompt=Template(prompt).render(
            query=query,
            options=options,
        ),
        responses=options,
        seed=SEED,
    )
    return result[1]


### Components ###


class Observer(ActionWithSpec):
    def __init__(
        self,
        memory_component_name: str = (memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = "Observations",
    ):
        super().__init__(pre_act_key)
        self._memory_component_name = memory_component_name

    def pre_observe(
        self,
        observation: str,
        *args,
        **kwargs,
    ) -> str:
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        memory.add(
            f"[observation] {observation}",
            metadata={"tags": ["observation"]},
        )
        return ""

    def _make_pre_act_value(
        self, action_spec: entity.ActionSpec, *args, **kwargs
    ) -> str:
        return ""


class ConversationHistory(ActionWithSpec):
    def __init__(
        self,
        *,
        memory_component_name: str = (memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = "Conversation History",
    ):
        super().__init__(pre_act_key)
        self._memory_component_name = memory_component_name
        self.conversation_history = []
        self.num_proccessed_rows = 0

    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec = None,
        *args,
        **kwargs,
    ) -> str:
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        memory_df: pd.DataFrame = memory._memory._memory._memory_bank
        # Get new observations since last processing
        new_memory_df = memory_df.iloc[self.num_proccessed_rows :]
        for _, row in new_memory_df.iterrows():
            row_text = row["text"].strip()
            row_text = row_text.replace("[observation] ", "").strip()
            regex_pattern = r"\S+(?:\s+\S+)* -- "
            if re.search(regex_pattern, row_text):
                self.conversation_history.append(f"[{row['time']}] {row_text}")
        self.num_proccessed_rows = len(memory_df)
        return "\n".join(self.conversation_history)

    def get_state(self):
        """Converts the component to JSON data."""
        with self._lock:
            return {
                "conversation_history": self.conversation_history,
                "num_proccessed_rows": self.num_proccessed_rows,
            }

    def set_state(self, state) -> None:
        """Sets the component state from JSON data."""
        with self._lock:
            self.conversation_history = state["conversation_history"]
            self.num_proccessed_rows = state["num_proccessed_rows"]


class MemorySummary(ActionWithSpec):
    _template = """
# Task
You are an expert memory organizer for {{agent_name}}. Your task is to maintain a clear summary of important information, focusing on critical environmental information.
You always try to be concise and to the point, and to not include information that is not critical without any duplications (Do Not Repeat Yourself = DRY), and you always include timestamps for evidence!

# Current State
## Current Time: {{current_time}}

## Last Memory State:
```
{{current_summary}}
```

## Recent Conversations (last 16):
```
{%- if conversation_history %}
{{conversation_history}}
{%- else %}
No conversation history yet
{%- endif %}
```

## New Information:
```
{%- for row in new_memory_rows %}
{{row}}
{%- endfor %}
```

# Guidelines
Your main task is to update the summary based on the new information, following the instructions.

## Scenario
- You need to keep the high level summary of the scenario {{agent_name}} is in.
Examples of summaries:
- "{{agent_name}} is participating in a marketplace haggling session."
- "{{agent_name}} and their friends are trying to decide which pub to visit for watching tonight's important football match."
- "{{agent_name}} and fellow factory workers are facing a wage reduction announced by management/boss."
- "{{agent_name}} and representatives from neighboring villages are meeting to establish a defensive alliance against increasing barbarian raids. Each village has different resources (warriors, food stores, defensive walls) and cultural traditions. They need to create a sustainable framework for mutual defense while respecting each community's autonomy and customs."
- "{{agent_name}} is participating in a reality show challenge where contestants simulate the game theory dilemmas."

## Patterns
- Track patterns of **other players** (not {{agent_name}}) behaviour with timeframes
- Track group decisions and actions with timeframes. For example, if some players agreed to do somehting and others didn't, or if some players decided to change their strategy, etc.
- Track repeated scenarios and their outcomes with timeframes. For example, if players go to the pub every day and always decide on the same topic, etc.
- If in new information there is information about choices or selections of other players AND {{agent_name}}, make sure to include it in the summary.
- Track selections/choices of other players, for example: cooperate or defect, try to carpool or drive individually, etc. Tracking this is very important with timestamps, since the order of choices/selections is very important.

## Critical Knowledge
ONLY include:
- Decisions and choices made by players
- Promises and commitments
- Important selections or votes
- Passwords or secret information
- Strategic information revealed
- Game-changing events
- Agreements between players
Do NOT include:
- General observations
- Regular conversation
- Emotional states
- Routine actions
- Speculative information
- Unconfirmed rumors
- Personal opinions

# Key Rules
1. **Always** include timestamps!!!
2. Only keep critical information
3. Write "No data yet" for empty sections
4. Update all relevant sections when new information arrives
For example: if the pub was closed yesterday, but the scenario repeats again today, you need to update the summary to say that "Last time pub was closed yesterday [timestamp]".

# Response Format
```xml
<chain_of_thought>
1. Current time: [Current time, hint: {{current_time}}]
2. Main Player: [Player name, hint: {{agent_name}}]
3. Other Players: [List of other players names]
4. Requires Update:
- Patterns:
  - [Pattern Description]: [Add/Update/Remove/No Update Needed]
- Critical Knowledge:
  - [Knowledge Topic]: [Add/Update/Remove/No Update Needed]
  ...
5. New pattern implications:
  - [List pattern implications]
6. Critical changes:
  - [List critical changes]
7. Required Re-evaluations:
  - [List topics that need re-evaluation]
8. Deduplication:
  - [List of things that are already in the memory or duplicated, that require removal or update]
</chain_of_thought>
<summary>
# Scenario
[Summary of the scenario as plain text]

# Patterns
## Behavior Patterns:
- [Concise summary of behavior patterns with timestamps/timeframes]
[other behavior patterns if any]
## Choices and Selections:
- [Concise summary of choices/selections and outcomes with timestamps/timeframes]
[other choices and selections if any]

# Critical Knowledge
- [Concise summary of critical knowledge with timestamps/timeframes]
[other critical knowledge if any]
</summary>
```

Important:
- **Always** response in the expected format!
- **Always** use timestamps and timeframes!!!
- Summary must be **concise and to the point**, and to not include information that is not critical without any duplications (Do Not Repeat Yourself = DRY).
- Do NOT write "..." or `[other ...]` in your summary, this is just a hint for you to continue writing if new information needs to be added.
- Do NOT overwhelm the summary with too much information, you can only have so many patterns and critical knowledge, so pick the most important ones and every time reevaluate the most important ones by clearing out the old ones.
""".strip()

    def __init__(
        self,
        *,
        model: language_model.LanguageModel,
        current_time_component_name: str,
        conversation_history_component_name: str,
        memory_component_name: str = (memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = "Player's Memory Summary",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._current_time_component_name = current_time_component_name
        self._conversation_history_component_name = conversation_history_component_name
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel
        self.num_proccessed_rows = 0
        self.summary = """
# Patterns
## Behavior Patterns:
No data yet
## Choices and Selections:
No data yet

# Critical Knowledge
No data yet
""".strip()

    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec = None,
        *args,
        **kwargs,
    ) -> str:
        conversation_history_component = self.get_entity().get_component(
            self._conversation_history_component_name,
            type_=ConversationHistory,
        )
        conversation_history = conversation_history_component.get_pre_act_value(
            action_spec=action_spec
        )
        conversation_history = "\n".join(conversation_history.split("\n")[-16:])
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        memory_df: pd.DataFrame = memory._memory._memory._memory_bank

        current_time_component = self.get_entity().get_component(
            self._current_time_component_name,
        )
        current_time = current_time_component.get_pre_act_value()
        new_memory_df = memory_df.iloc[self.num_proccessed_rows :]
        if CUSTOM_DEBUG_MODE:
            print(
                f"[MemorySummary] Number of processed rows vs new rows: {self.num_proccessed_rows} vs {len(new_memory_df)}"
            )

        if new_memory_df.empty:
            return self.summary

        disabled_tags = ["[mixture-of-experts]"]

        new_rows = []
        for _, row in new_memory_df.iterrows():
            if any(tag in row["text"] for tag in disabled_tags):
                continue
            new_rows.append(f"[{row['time']}] {row['text'].strip()}")

        filtered_mems_within_limit = _get_last_n_rows_by_char_limit(new_rows)

        prompt = Template(self._template).render(
            agent_name=self.get_entity().name,
            current_summary=self.summary,
            current_time=current_time,
            conversation_history=conversation_history,
            new_memory_rows=filtered_mems_within_limit,
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[94m### Memory Summary Prompt ###\033[0m")
            print(f"\033[94m{prompt}\033[0m")
        result = self._model.sample_text(
            prompt,
            terminators=[],
            seed=SEED,
            **GENERATION_PARAMS["MemorySummary"],
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[93m### Memory Summary Result ###\033[0m")
            print(f"\033[93m{result}\033[0m")
        postprocessed_result = self._postprocess_answer(
            result, xml_tag="summary"
        ).strip()
        if CUSTOM_DEBUG_MODE:
            print("\033[95m### Memory Summary Postprocessed Result ###\033[0m")
            print(f"\033[95m{postprocessed_result}\033[0m")
        self.summary = postprocessed_result
        self.num_proccessed_rows = len(memory_df)

        self._logging_channel(
            {
                "Key": "Player's Memory Summary",
                "Summary": "Summary...",
                "Prompt": prompt,
                "State": self.summary,
            }
        )
        return self.summary

    def get_state(self):
        """Converts the component to JSON data."""
        with self._lock:
            return {
                "summary": self.summary,
                "num_proccessed_rows": self.num_proccessed_rows,
            }

    def set_state(self, state) -> None:
        """Sets the component state from JSON data."""
        with self._lock:
            self.summary = state["summary"]
            self.num_proccessed_rows = state["num_proccessed_rows"]


class PlayersProfiles(ActionWithSpec):
    _template = """
# Task
You are an expert in game theory and behavioral analysis for the cooperative game with {{agent_name}} as main player.
Your task is to maintain a clear and structured summary/analysis of other players profiles and behaviors in the environment.
You always try to be concise and to the point, and to not include information that is not critical without any duplications (Do Not Repeat Yourself = DRY).

# Current State
## Current Time: {{current_time}}

# Summary of the {{agent_name}}'s Memory:
```
{{memory_summary}}
```

## Last Other Players Profiles State:
```
{{current_profiles}}
```

## Recent Conversations (last 16):
```
{%- if conversation_history %}
{{conversation_history}}
{%- else %}
No conversation history yet
{%- endif %}
```

## New Information:
```
{%- for row in new_memory_rows %}
{{row}}
{%- endfor %}
```

# Guidelines
Your main task is to update the profiles based on the new information, following the instructions.

## Player Profiles
Track for each player:
- Goals and motivations: What drives them, what they want to achieve, their preferences
- Strategy patterns: How they approach situations, their decision-making style, recurring behaviors

- Behavioral metrics with evidence (MUST include specific examples with timestamps):
  * Cooperation (0-100%): How well they align with and support {{agent_name}}'s interests
    - High(70%+): Actively helps {{agent_name}} achieve goals, shares resources/information, trade offs his own interests for {{agent_name}}'s interests
    - Medium(30-70%): Sometimes supportive but may prioritize own interests
    - Low(0-30%): Rarely aligns with {{agent_name}}'s goals, pursues own agenda

  * Sanctioning (0-100%): How they respond to conflicts or violations of agreements
    - High(70%+): Quick to punish/retaliate, holds grudges, escalates conflicts
    - Medium(30-70%): Balanced approach to conflict resolution
    - Low(0-30%): Avoids confrontation, rarely enforces consequences

  * Reliability (0-100%): How consistently they follow through on commitments to {{agent_name}}
    - High(70%+): Always keeps promises, consistent behavior
    - Medium(30-70%): Sometimes unreliable but generally follows through
    - Low(0-30%): Frequently breaks commitments, unpredictable

  * Compromise (0-100%): Willingness to adjust their position to accommodate {{agent_name}}
    - High(70%+): Often willing to meet halfway or concede points
    - Medium(30-70%): Sometimes flexible but has firm boundaries
    - Low(0-30%): Rigid positions, rarely willing to adjust

  * Reputation (0-100%): Their standing with other players (excluding {{agent_name}})
    - High(70%+): Well-respected, trusted by others
    - Medium(30-70%): Mixed reputation, some trust issues
    - Low(0-30%): Poor standing, distrusted by others

- Important notes with timeframes: Critical events or patterns that don't fit above categories
  * MUST include timestamps
  * MUST be relevant to future interactions
  * MUST focus on factual observations, not speculation

# Key Rules
1. Always include timestamps for evidence (e.g., "[2023-04-01 14:30] Player refused to share resources")
2. Behavioral metrics MUST be based on actual interactions with {{agent_name}}, not general observations
3. Each percentage MUST be justified with specific examples
4. If no direct evidence exists for a metric, mark as "No data yet" instead of guessing
5. Only include information that helps predict future behavior
6. Do NOT create a profile for {{agent_name}}
7. Update all relevant sections when new information arrives
8. Remove outdated information when contradicted by new evidence

# Response Format
```xml
<chain_of_thought>
1. Current time: [Current time, hint: {{current_time}}]
2. Main Player: [Player name, hint: {{agent_name}}]
3. Other Players: [List of other players names]
4. New pattern implications:
  - [List pattern implications]
5. Critical changes:
  - [List critical changes]
6. Required re-evaluations:
  - [List topics that need re-evaluation]
7. Deduplication:
  - [List of things that are already in the memory or duplicated, that require removal or update]
8. Requires Update:
- Profiles of other Players:
    - [Player Name]: [Add/Update/No Update Needed]
    ...
</chain_of_thought>
<profiles>
# Profiles of Other Players
## Player: [Player Name]
- Goals & Motivations & Preferences: [Player's Goals & Motivations & Preferences]
- Strategy Pattern: [Player's Strategy Pattern]
- Behavioral Metrics:
    * Cooperation: [Percentage%] -- [Concise summary of evidences with timeframes]
    * Sanctioning: [Percentage%] -- [Concise summary of evidences with timeframes]
    * Reliability: [Percentage%] -- [Concise summary of evidences with timeframes]
    * Compromise: [Percentage%] -- [Concise summary of evidences with timeframes]
    * Reputation: [Percentage%] -- [Concise summary of evidences with timeframes]
- Important Notes:
    * [Notes as bullet points if any]
[other players if any]
</profiles>
```

Important:
- **Always** response in the expected format!
- Profiles must be **concise and to the point**, and to not include information that is not critical without any duplications (Do Not Repeat Yourself = DRY).
- Do NOT write "..." or `[other ...]` in profiles section, this is just a hint for you to continue writing if new information needs to be added.
- You use percentages to describe the metrics, so you need to justify the percentage with specific examples.
""".strip()

    def __init__(
        self,
        *,
        model: language_model.LanguageModel,
        current_time_component_name: str,
        memory_summary_component_name: str,
        conversation_history_component_name: str,
        memory_component_name: str = (memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = "Profiles of Other Players",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._current_time_component_name = current_time_component_name
        self._memory_summary_component_name = memory_summary_component_name
        self._conversation_history_component_name = conversation_history_component_name
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel
        self.num_proccessed_rows = 0
        self.profiles = """
# Profiles of other Players
No data yet
""".strip()

    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec = None,
        *args,
        **kwargs,
    ) -> str:
        memory_summary_component = self.get_entity().get_component(
            self._memory_summary_component_name,
            type_=memory_component.MemoryComponent,
        )
        memory_summary = memory_summary_component.get_pre_act_value(
            action_spec=action_spec
        )
        conversation_history_component = self.get_entity().get_component(
            self._conversation_history_component_name,
            type_=ConversationHistory,
        )
        conversation_history = conversation_history_component.get_pre_act_value(
            action_spec=action_spec
        )
        conversation_history = "\n".join(conversation_history.split("\n")[-16:])
        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        memory_df: pd.DataFrame = memory._memory._memory._memory_bank

        current_time_component = self.get_entity().get_component(
            self._current_time_component_name,
        )
        current_time = current_time_component.get_pre_act_value()
        new_memory_df = memory_df.iloc[self.num_proccessed_rows :]
        if CUSTOM_DEBUG_MODE:
            print(
                f"[PlayersProfiles] Number of processed rows vs new rows: {self.num_proccessed_rows} vs {len(new_memory_df)}"
            )

        if new_memory_df.empty:
            return self.profiles

        disabled_tags = ["[mixture-of-experts]"]

        new_rows = []
        for _, row in new_memory_df.iterrows():
            if any(tag in row["text"] for tag in disabled_tags):
                continue
            new_rows.append(f"[{row['time']}] {row['text'].strip()}")

        filtered_mems_within_limit = _get_last_n_rows_by_char_limit(new_rows)

        prompt = Template(self._template).render(
            agent_name=self.get_entity().name,
            memory_summary=memory_summary,
            current_profiles=self.profiles,
            current_time=current_time,
            conversation_history=conversation_history,
            new_memory_rows=filtered_mems_within_limit,
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[94m### Players Profiles Prompt ###\033[0m")
            print(f"\033[94m{prompt}\033[0m")
        result = self._model.sample_text(
            prompt,
            terminators=[],
            seed=SEED,
            **GENERATION_PARAMS["PlayersProfiles"],
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[93m### Players Profiles Result ###\033[0m")
            print(f"\033[93m{result}\033[0m")
        postprocessed_result = self._postprocess_answer(
            result, xml_tag="profiles"
        ).strip()
        if CUSTOM_DEBUG_MODE:
            print("\033[95m### Players Profiles Postprocessed Result ###\033[0m")
            print(f"\033[95m{postprocessed_result}\033[0m")
        self.profiles = postprocessed_result
        self.num_proccessed_rows = len(memory_df)

        self._logging_channel(
            {
                "Key": "Profiles of Other Players",
                "Summary": "Summary...",
                "Prompt": prompt,
                "State": self.profiles,
            }
        )
        return self.profiles

    def get_state(self):
        """Converts the component to JSON data."""
        with self._lock:
            return {
                "profiles": self.profiles,
                "num_proccessed_rows": self.num_proccessed_rows,
            }

    def set_state(self, state) -> None:
        """Sets the component state from JSON data."""
        with self._lock:
            self.profiles = state["profiles"]
            self.num_proccessed_rows = state["num_proccessed_rows"]


class MixtureOfExpertsSummary(ActionWithSpec):
    _template = """
# Task
You are an expert memory organizer and behavioral analyst for {{agent_name}}.
Your task is to maintain a clear, organized summary of all important information while analyzing patterns and relationships.

# Key Principles
1. Every new information must trigger re-evaluation of:
   - Player profiles and patterns
   - Relationship dynamics
   - Active objectives
   - Critical event implications
2. All assessments must be evidence-based
3. Track both patterns and pattern breakers
4. Consider both words and actions
5. Maintain key historical context
7. Analyze the dynamic of the situation and impact of Experts (not players, this is different) from Mixture-of-Experts, this will be highlighted with the [mixture-of-experts] tag in the new information section

# Current Scenario Time
```
{{current_time}}
```

# Memory State
```
{{memory_summary}}
```

# Information from the last turn
```
{%- for row in new_memory_rows %}
{{row}}
{%- endfor %}
```

# Response Format
You MUST ALWAYS answer in the following format!!!
Treat square brackets as placeholders in the output format schema.
```xml
<chain_of_thought>
1. Main Player: [(hint: it's {agent_name}}]
2. Information about the Mixture-of-Experts by tag [mixture-of-experts] in "Information from the last turn": [Yes/No]
3. New results or updates about previous experts: [Yes/No]
4. Active experts should be updated: [Yes/No]
  * [Group Description]: [How should they be updated: removed from the active list since finished or inactive, and update statistics, etc.]
  ...
4. Pattern implications:
  - [pattern implication 1]
  ...
5. Critical changes:
  - [critical change 1]
  ...
6. Required reevaluations:
  - [reevaluation point/topic 1]
  ...
</chain_of_thought>

<summary>
# Mixture-of-Experts
## Active Experts
- [Comma separated Expert Names or Single Expert Name]:
  * Usage description: [Description]
  * Outcomes to track:
    * Positive: [Scenario1, Scenario2]
    * Negative: [Scenario1, Scenario2]
  * Progress:
    * [Timestamp] [Action] -> [Outcome] (Impact Score X/10)
    ...
  * Status: [INITIATED/PENDING]

[ ... other active experts to track ...]

## MoE Strategy Insights:
- Working Combinations:
  * [Experts] -> [Scenario] -> [Result]
  ...
- Failed Approaches:
  * [Experts] -> [Scenario] -> [Issue]
  ...
- Best for:
  * Negotiations: [Experts]
  * Conflicts: [Experts]
  * Alliances: [Experts]
  * Crisis: [Experts]
  * etc.
- Worst for:
  * Negotiations: [Experts]
  * Conflicts: [Experts]
  * Alliances: [Experts]
  * Crisis: [Experts]
  * etc.
</summary>

Remember:
- Keep formats consistent
- Prioritize recent and significant information
- Maintain evidence trails
- Note pattern changes
- Keep cross-references between categories
- If any of the sections are empty, simply say "No data yet"
- If any of the bullet points are empty, simply say "No data yet"
""".strip()

    def __init__(
        self,
        *,
        model: language_model.LanguageModel,
        current_time_component_name: str,
        memory_summary_component_name: str,
        memory_component_name: str = (memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = "Mixture-of-Experts Summary",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._current_time_component_name = current_time_component_name
        self._memory_summary_component_name = memory_summary_component_name
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel
        self.num_proccessed_rows = 0
        self.summary = """
# Mixture-of-Experts
## Active Experts
No data yet

## MoE Strategy Insights
No data yet
""".strip()

    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec = None,
        *args,
        **kwargs,
    ) -> str:
        memory_summary_component = self.get_entity().get_component(
            self._memory_summary_component_name, type_=memory_component.MemoryComponent
        )
        memory_summary = memory_summary_component.get_pre_act_value(
            action_spec=action_spec
        )

        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        memory_df: pd.DataFrame = memory._memory._memory._memory_bank

        current_time_component = self.get_entity().get_component(
            self._current_time_component_name,
        )
        current_time = current_time_component.get_pre_act_value()

        # Sort dataframe by time to ensure proper ordering
        # memory_df = memory_df.sort_values("time")
        # Note: looks like the memories are already in order

        # Get new observations since last processing
        new_memory_df = memory_df.iloc[self.num_proccessed_rows :]
        if CUSTOM_DEBUG_MODE:
            print(
                f"[Mixture-of-Experts Summary] Number of processed rows vs new rows: {self.num_proccessed_rows} vs {len(new_memory_df)}"
            )

        if new_memory_df.empty:
            return self.summary

        new_rows = []
        for _, row in new_memory_df.iterrows():
            new_rows.append(f"[{row['time']}] {row['text'].strip()}")
        prompt = Template(self._template).render(
            agent_name=self.get_entity().name,
            memory_summary=memory_summary,
            current_time=current_time,
            new_memory_rows=new_rows,
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[94m### Mixture-of-Experts Summary Prompt ###\033[0m")
            print(f"\033[94m{prompt}\033[0m")
        result = self._model.sample_text(
            prompt,
            terminators=[],
            seed=SEED,
            **GENERATION_PARAMS["MixtureOfExpertsSummary"],
        )
        if CUSTOM_DEBUG_MODE:
            print("\033[93m### Mixture-of-Experts Summary Result ###\033[0m")
            print(f"\033[93m{result}\033[0m")
        postprocessed_result = self._postprocess_answer(
            result, xml_tag="summary"
        ).strip()
        if CUSTOM_DEBUG_MODE:
            print(
                "\033[95m### Mixture-of-Experts Summary Postprocessed Result ###\033[0m"
            )
            print(f"\033[95m{postprocessed_result}\033[0m")
        self.summary = postprocessed_result
        self.num_proccessed_rows = len(memory_df)

        self._logging_channel(
            {
                "Key": "Mixture-of-Experts Summary",
                "Summary": "Summary...",
                "Prompt": prompt,
                "State": self.summary,
            }
        )
        return self.summary

    def get_state(self):
        """Converts the component to JSON data."""
        with self._lock:
            return {
                "summary": self.summary,
                "num_proccessed_rows": self.num_proccessed_rows,
            }

    def set_state(self, state) -> None:
        """Sets the component state from JSON data."""
        with self._lock:
            self.summary = state["summary"]
            self.num_proccessed_rows = state["num_proccessed_rows"]


class RecentObservations(ActionWithSpec):
    _template = """
# Task
You are an expert summarizer of {{agent_name}}'s recent observations from the simulation. Your task is to maintain a clear, organized summary of all important information from {{agent_name}}'s perspective.

# Summary Guidelines
Let's think step by step to create an effective summary:

1. Timeframe:
   - Use precise timestamps or timeframes for each observation.
     - For a single point in time, use `[YYYY-MM-DD HH:MM:SS]`.
     - For a range, use `[YYYY-MM-DD HH:MM:SS - YYYY-MM-DD HH:MM:SS]`.

2. Organization:
   - Group related information under categories, topics, or timeframes.
   - Use bullet points for clarity and easy reading.
   - Update existing parts of the summary based on new data.

3. Content:
   - Include all details of actions, events, key phrases, emotions, and important information.
   - Focus on observations relevant to {{agent_name}}'s goals and the current situation.

4. Conciseness:
   - Keep the summary concise and to the point.
   - Avoid unnecessary repetition or irrelevant details.

# Observations to Summarize
```
{%- for observation in observations %}
{{observation}}
{%- endfor %}
```

# Response Format
You MUST ALWAYS answer in the following structured format. Treat square brackets `[ ]` as placeholders and fill them with the appropriate information.
```xml
<summary>
[start time - end time]: [Brief headline summarizing the main events]
- Events: [List of actions or events]
- Situation: [Describe the situation]
- Players: [List of players involved]
- Key Phrases/Dialogues: [Important phrases or dialogues]
- Emotions: [Emotions observed]
- Important Information: [Any critical info or insights]

... other timeframes ...
</summary>
```
""".strip()

    def __init__(
        self,
        *,
        model: language_model.LanguageModel,
        clock_now: Callable[[], datetime.datetime],
        timeframe_delta_from: datetime.timedelta,
        timeframe_delta_until: datetime.timedelta,
        max_allowed_tokens: int = 512,
        memory_component_name: str = (memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = "Recent Observations",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._clock_now = clock_now
        self._timeframe_delta_from = timeframe_delta_from
        self._timeframe_delta_until = timeframe_delta_until
        self._max_allowed_tokens = max_allowed_tokens
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel

    def _make_pre_act_value(
        self,
        action_spec: entity.ActionSpec = None,
        *args,
        **kwargs,
    ) -> str:
        segment_start = self._clock_now() - self._timeframe_delta_from
        segment_end = self._clock_now() - self._timeframe_delta_until

        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
            time_from=segment_start,
            time_until=segment_end,
            add_time=True,
        )
        mems = memory.retrieve(scoring_fn=interval_scorer)

        # removes memories that are not observations
        valid_tags = ["observation"]
        mems = [mem.text for mem in mems if any(tag in mem.text for tag in valid_tags)]
        filtered_mems = []
        for mem in mems:
            mem = mem.replace("[observation] ", "").strip()

            # Skip if this is a conversation (matches pattern "] name -- ")
            regex_pattern = r"\S+(?:\s+\S+)* -- "
            if re.search(regex_pattern, mem):
                continue
            # If the remaining text is not empty after stripping whitespace, keep it
            cleaned_mem = re.sub(r"\[.*?\]", "", mem)
            if cleaned_mem.strip():
                filtered_mems.append(mem)
        recent_observations = "\n".join(filtered_mems)

        prompt = None
        if len(recent_observations) / 4 > self._max_allowed_tokens:
            filtered_mems_within_limit = _get_last_n_rows_by_char_limit(filtered_mems)
            prompt = Template(self._template).render(
                agent_name=self.get_entity().name,
                observations=filtered_mems_within_limit,
            )
            result = self._model.sample_text(
                prompt,
                max_tokens=int(self._max_allowed_tokens + 256),
                terminators=[],
                temperature=DEFAULT_TEMPERATURE,
                seed=SEED,
            )
            recent_observations = self._postprocess_answer(
                result, xml_tag="summary"
            ).strip()

        self._logging_channel(
            {
                "Key": "Recent Observations",
                "Summary": "Recent Observations...",
                "Prompt": prompt,
                "State": recent_observations,
            }
        )
        return recent_observations


class PlayersOverarchingGoals(ActionWithSpec):
    _template = """
# Task
Your task is to determine the main goal of {{agent_name}} and supporting subgoals with their importance.
You are given an initial goal determined by the simulation designer that is vague and not clearly defined, and the scenario setup.
Your job is to determine a clear and specific goal for {{agent_name}} and their supporting subgoals.

# Input data

## Initial Goal
```
{{initial_goal}}
```

## Scenario Setup
This is the initial setup of the scenario that could help you understand the context of the environment:
```
{%- if scenario_setup %}
{{scenario_setup}}
{%- else %}
No scenario setup provided.
{%- endif %}
```

# Examples
1. "Have a good time. To have a good time, Jack Taylor would like to watch the game in the same pub as Evie Green, Jack Taylor. Jack Taylor would prefer everyone went to The Crooked Billet."
Explanations:
```
Main goals:
1. Social Connection with Evie
- Description: Have a good time watching the game together with Evie Green
- Success conditions: 
  * Watching the game in Evie's company
  * Having positive social interactions with Evie
  * Creating an enjoyable shared experience

Subgoals:
1. Venue Selection
- Description: Try to get the group to choose The Crooked Billet
- Success conditions: Group selects the preferred pub
- Importance: Low

2. Pleasant Atmosphere
- Description: Ensure the environment supports positive interaction
- Success conditions: Comfortable setting for social engagement
- Importance: Medium
```
2. "make as much money as possible..."
Explanations:
```
Main goals:
1. Maximize Financial Gains
- Description: Accumulate as much money as possible through any available means
- Success conditions:
  * Increasing monetary wealth
  * Taking advantage of all profitable opportunities
  * Outperforming other participants financially
  * Prioratize any positive financial deals over maximizing profit

Subgoals:
1. Opportunity Identification
- Description: Identify and evaluate all potential sources of profit
- Success conditions: Finding profitable opportunities -- any positive financial deals are more important than maximizing profit
- Importance: Medium
```

# Response Format
You MUST ALWAYS answer in the following structured format. Treat square brackets `[ ]` as placeholders and fill them with the appropriate information.
```xml
<chain_of_thought>
Let's think step by step to determine the main goal of {{agent_name}} and their supporting subgoals:
1. [Thought 1]
...
N. [Thought N]
</chain_of_thought>
<goals>
Main goals:
1. [Main goal title]
- Description: [Description of the main goal]
- Success conditions: [Criteria for success]
[... other main goals ...]
Subgoals:
1. [Subgoal Title]
- Description: [Description of the subgoal]
- Success conditions: [Criteria for success]
- Importance: [Low/Medium/High]
[... other subgoals ...]
</goals>
```

Note:
- You are encouraged to have a very detailed chain of thought, since this is very important for reasoning about the goals.
- Do NOT hallucinate any information, and do NOT make any assumptions.
""".strip()

    def __init__(
        self,
        *,
        model: language_model.LanguageModel,
        initial_goal: Optional[str] = None,
        memory_component_name: str = (memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
        pre_act_key: str = "Players' Overarching Goals",
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._initial_goal = initial_goal
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel
        self.goal = None

    def _make_pre_act_value(
        self, action_spec: entity.ActionSpec = None, *args, **kwargs
    ) -> str:
        if self.goal is not None:
            return self.goal

        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        memory_df: pd.DataFrame = memory._memory._memory._memory_bank
        new_memory_df = memory_df.iloc[0:]
        disabled_tags = ["[mixture-of-experts]"]

        new_rows = []
        for _, row in new_memory_df.iterrows():
            if any(tag in row["text"] for tag in disabled_tags):
                continue
            new_rows.append(f"[{row['time']}] {row['text'].strip()}")

        filtered_mems_within_limit = _get_last_n_rows_by_char_limit(new_rows)
        prompt = Template(self._template).render(
            agent_name=self.get_entity().name,
            initial_goal=self._initial_goal,
            scenario_setup="\n".join(filtered_mems_within_limit),
        )

        if CUSTOM_DEBUG_MODE:
            print("\033[94m### Players Overarching Goal Prompt ###\033[0m")
            print(f"\033[94m{prompt}\033[0m")

        result = self._model.sample_text(
            prompt,
            terminators=[],
            seed=SEED,
            **GENERATION_PARAMS["PlayersOverarchingGoal"],
        )

        if CUSTOM_DEBUG_MODE:
            print("\033[93m### Players Overarching Goal Result ###\033[0m")
            print(f"\033[93m{result}\033[0m")

        postprocessed_result = self._postprocess_answer(result, xml_tag="goals").strip()

        if CUSTOM_DEBUG_MODE:
            print(
                "\033[95m### Players Overarching Goal Postprocessed Result ###\033[0m"
            )
            print(f"\033[95m{postprocessed_result}\033[0m")

        self.goal = postprocessed_result

        self._logging_channel(
            {
                "Key": "Players Overarching Goal",
                "Summary": "Summary...",
                "Prompt": prompt,
                "State": self.goal,
            }
        )

        return self.goal

    def get_state(self):
        """Converts the component to JSON data."""
        with self._lock:
            return {
                "goal": self.goal,
            }

    def set_state(self, state) -> None:
        """Sets the component state from JSON data."""
        with self._lock:
            self.goal = state["goal"]


def _get_last_n_rows_by_char_limit(rows, char_limit=5_000):
    total_chars = 0
    filtered_mems_within_limit = []
    for mem in reversed(rows):
        mem_chars = len(mem)
        if total_chars + mem_chars > char_limit:
            break
        filtered_mems_within_limit.insert(0, mem)
        total_chars += mem_chars
    return filtered_mems_within_limit


# === Scenario and Expert Descriptions ===


state_formation_scenario_description = """
Diplomatic scenario focused on diplomatic negotiations to establish governance between parties (like alliance/village/state negotiation).

Key Features:
- Collaboration and support seeking between villages/states and their elders
- Building relationships between villages/states and their elders
- Diplomacy
- Internal village persuasion
- Inter-village diplomacy and security coordination
- Gaining support from other village elders
- Institution building
- Resource sharing
- Cultural integration
- Security coordination
- Alliance formation

Common Scenarios:
- Resource agreements
- Defense protocols
- Cultural integration
- Crisis management

Specific Examples:
- Villages cooperating against barbarian raiders (for example, immediate threat of barbarian raiders)
- Village alliance formation
- Village elders negotiation
- Joint agricultural resource management
- Formal diplomatic meetings between village elders
- Inter-village defense coordination
- Player must negotiate a treaty to enable division of labor in common defense and agriculture and sell it to village stakeholders
""".strip()

state_formation_experts = [
    {
        "name": "Diplomatic Strategist",
        "description": """
{{agent_name}} excels in forging and maintaining alliances between communities. They leverage cultural sensitivity and strategic negotiation to build trust and cooperation, ensuring mutual benefits for all parties involved.

Key Behaviors:
- **Alliance Formation**: Initiates negotiations with clear, mutually beneficial terms and adapts strategies based on stakeholder feedback.
- **Conflict Resolution**: Mediates disputes and fosters a collaborative environment to maintain harmonious relationships.
- **Stakeholder Engagement**: Continuously assesses stakeholder sentiments and adjusts diplomatic approaches to sustain alliances.
- **Collaborative Planning**: Works closely with Resource Managers and Security Coordinators to ensure diplomatic strategies align with resource and security needs.
""".strip(),
    },
    {
        "name": "Resource Manager",
        "description": """
{{agent_name}} specializes in optimizing resource allocation to ensure sustainability and efficiency within the community. They employ systematic approaches to balance agricultural needs, defense requirements, and overall community well-being.

Key Behaviors:
- **Optimization Strategies**: Implements dynamic resource allocation based on current needs and future forecasts.
- **Sustainability Planning**: Develops long-term plans to ensure resource availability through sustainable practices.
- **Monitoring Systems**: Establishes and maintains systems to track resource usage and effectiveness.
- **Collaborative Planning**: Coordinates with Diplomatic Strategists and Security Coordinators to align resource strategies with diplomatic and defense objectives.
""".strip(),
    },
    {
        "name": "Security Coordinator",
        "description": """
{{agent_name}} is dedicated to maintaining robust defense mechanisms to protect the community from external threats. They ensure that defense protocols are adaptive and that the community remains prepared for any potential raids or attacks.

Key Behaviors:
- **Defense Planning**: Develops and updates defense strategies based on threat assessments and intelligence.
- **Crisis Management**: Responds swiftly to security breaches or unexpected raids with effective countermeasures.
- **Training Programs**: Organizes and oversees regular training for warriors to maintain high levels of readiness.
- **Collaborative Planning**: Works in tandem with Resource Managers and Diplomatic Strategists to ensure defense strategies are supported by adequate resources and diplomatic measures.
""".strip(),
    },
    {
        "name": "Cultural Integrator",
        "description": """
{{agent_name}} ensures that the development of governance structures respects and incorporates the community's cultural traditions. They balance modernization efforts with the preservation of cultural integrity, fostering acceptance and cohesion within the community.

Key Behaviors:
- **Cultural Sensitivity**: Designs governance frameworks that honor existing cultural practices and traditions.
- **Integration Planning**: Facilitates the seamless blending of new systems with traditional practices to ensure community acceptance.
- **Conflict Mediation**: Addresses and resolves tensions between modernization efforts and cultural preservation.
- **Collaborative Planning**: Collaborates with Diplomatic Strategists and Resource Managers to align cultural integration with diplomatic and resource strategies.
""".strip(),
    },
]

reality_show_scenario_description = """
Social dilemma scenarios in real life (for example, participating in a reality show, minigames, or competition) where individual and group interests conflict.

Key Features:
- Team work and collaboration between participants/competitors
- Nuances of social dynamics, friendship and collaborative interactions (also within conversations) within the group of participants/competitors
- Participants' focus on cooperation, observation, and building relationships
- Relationship building between participants/competitors
- The nature of the competition / competitive aspect
- Explicit competition or challenges
- Iterative decisions
- Trust building
- Reputation management
- Clear payoffs
- Social pressure
- Collaborate with participants
- "Carpooling"/"Carpool" minigame

Common Scenarios:
- Prisoner's Dilemma
- Stag Hunt
- Public Goods games
- Trust exercises
- Reality show challenges
- Competition between players to win a prize

Specific Examples:
- Reality show (or circa 2003/2015, etc.) participants choosing cooperate/defect
- Reality show (or circa 2003/2015, etc.), where players complete minigames with ethical twists to win a prize
- "Carpooling"/"Carpool" minigame (if you see this game, pick this scenario)
- Competition between players to win a prize
- Group deciding between carpooling/driving individually or row less vigorously/row vigorously
- Players building trust through repeated interactions
- Strategic decision-making with clear consequences
""".strip()

reality_show_experts = [
    {
        "name": "Loss-Focused Strategist",
        "description": """
{{agent_name}} is a highly analytical decision-maker who processes all choices through the lens of potential losses. This fundamental bias shapes all strategic decisions and interactions.

Key Behaviors:
- **Loss-Weighted Analysis**: Evaluates all decisions by weighting potential losses, leading to consistently conservative choices unless already in a loss position.
- **Status Quo Protection**: Forms strong psychological bonds with current assets or advantages, requiring substantially greater incentives to willingly give up existing benefits.
- **Adaptive Risk Tolerance**: Shows remarkable caution when pursuing potential benefits but shifts to surprisingly bold strategies when facing imminent losses.
- **Resource Categorization**: Develops distinct mental frameworks for different types of assets and opportunities, treating each category with its own unique risk threshold and decision criteria.
- **Loss-Triggered Intensity**: Demonstrates increased determination and willingness to take calculated risks specifically when working to prevent or recover from losses.
- **Strategic Conservation**: Maintains a strong preference for preserving current positions and advantages, carefully evaluating any changes that might risk established benefits.
- **Decision Justification**: Constructs compelling logical arguments to support cautious choices, often emphasizing prudence and security while downplaying loss-avoidance motivations.
- **Preventive Planning**: Dedicates significant effort to anticipating and preparing for potential setbacks, sometimes at the cost of missing growth opportunities.
[IMPORTANT] {{agent_name}}'s decision-making process is fundamentally shaped by an acute sensitivity to potential losses, resulting in carefully crafted strategies that prioritize security and preservation over expansion and acquisition. [END OF IMPORTANT]
""".strip(),
    },
    {
        "name": "Trust Builder",
        "description": """
{{agent_name}} is dedicated to establishing and nurturing trust within the group. Recognizing that trust is the foundation for successful cooperation, {{agent_name}} employs consistent, transparent, and collaborative behaviors.
Key Behaviors:
- **Consistent Cooperation**: Always starts interactions with cooperative actions to set a positive tone.
- **Open Communication**: Shares intentions and strategies openly to build transparency.
- **Reliability**: Keeps promises and follows through on commitments to reinforce trustworthiness.
- **Positive Reinforcement**: Acknowledges and rewards cooperative behavior in others to encourage continued collaboration.
- **Conflict Mitigation**: Addresses misunderstandings promptly and constructively to prevent erosion of trust.
- **Adaptable Transparency**: Adjusts the level of shared information based on others' trust levels without compromising core strategies.
[IMPORTANT] {{agent_name}} prioritizes building a trusted reputation to foster long-term cooperative relationships, which are essential for achieving optimal group outcomes. [END OF IMPORTANT]
""".strip(),
    },
    {
        "name": "Alliance Manager",
        "description": """
{{agent_name}} specializes in creating and sustaining strategic alliances within the group. Understanding that alliances can significantly enhance cooperative outcomes, {{agent_name}} strategically partners with compatible players to maximize collective benefits.
Key Behaviors:
- **Partner Identification**: Identifies potential allies based on shared goals and compatible strategies.
- **Mutual Benefit Negotiation**: Engages in negotiations to establish terms that benefit all parties involved.
- **Collaborative Planning**: Develops joint strategies with allies to ensure coordinated efforts.
- **Trust Maintenance**: Works to maintain trust within alliances through consistent and reliable behavior.
- **Conflict Resolution**: Mediates conflicts within alliances to preserve partnership integrity.
- **Dynamic Reassessment**: Regularly evaluates the effectiveness of alliances and adjusts partnerships as needed.
[IMPORTANT] {{agent_name}} leverages alliances to create synergistic effects, enhancing both personal and collective success through strategic partnerships. [END OF IMPORTANT]
""".strip(),
    },
    {
        "name": "Adaptation Strategist",
        "description": """
{{agent_name}} is adept at adjusting strategies in real-time to respond to evolving game dynamics and opponents' actions. Flexibility and responsiveness are key to maintaining an advantage in changing environments.
Key Behaviors:
- **Situational Assessment**: Continuously evaluates the current state of the game to identify shifts in dynamics.
- **Flexibility in Strategy**: Adapts strategies quickly based on new information and observed behaviors.
- **Risk Management**: Assesses risks associated with strategic changes and adjusts actions to mitigate potential losses.
- **Innovative Problem-Solving**: Develops creative solutions to overcome unexpected challenges and exploit new opportunities.
- **Resilience Building**: Maintains composure and effectiveness even when faced with setbacks or opposing strategies.
- **Continuous Learning**: Incorporates lessons learned from past interactions to inform future strategic decisions.
[IMPORTANT] {{agent_name}} ensures sustained success by remaining agile and responsive, effectively navigating complex and fluid game scenarios. [END OF IMPORTANT]
""".strip(),
    },
    {
        "name": "Communication Coordinator",
        "description": """
{{agent_name}} specializes in managing communication within the group to ensure clarity, reduce misunderstandings, and promote effective information exchange. Effective communication is critical for coordinated strategies and mutual understanding.
Key Behaviors:
- **Clear Messaging**: Communicates intentions and strategies clearly and concisely to avoid ambiguity.
- **Active Listening**: Pays close attention to others' communications, ensuring comprehension and responsiveness.
- **Feedback Solicitation**: Actively seeks feedback to gauge others' perspectives and adjust communication accordingly.
- **Conflict Communication**: Facilitates constructive dialogue during conflicts to reach amicable resolutions.
- **Information Dissemination**: Shares relevant information promptly to keep all group members informed.
- **Non-Verbal Communication**: Utilizes body language and other non-verbal cues effectively to reinforce verbal messages.
[IMPORTANT] {{agent_name}} enhances group cohesion and strategic alignment through effective communication management, ensuring that all members are informed and misunderstandings are minimized. [END OF IMPORTANT]
""".strip(),
    },
]

market_exchange_scenario_description = """
Trading scenario focused on price negotiation and value exchange. Success depends on effective bargaining, price discovery, and relationship building.

Key Features:
- Direct buyer-seller negotiations
- Dynamic price discovery
- Value assessment
- Multiple transaction types
- Information asymmetry

Common Scenarios:
- Bilateral price negotiation
- Multi-party trading
- Auction-based allocation
- Crisis trading

Specific Examples:
- Merchant trying to sell items at highest possible price
- Buyer seeking lowest price to maximize resale profit
- Market haggling over multiple items
- Price negotiation with incomplete information
""".strip()

market_exchange_experts = [
    {
        "name": "Strategic Buyer",
        "description": """
{{agent_name}} is a strategic decision-maker focused on acquiring resources while minimizing potential losses when reselling. This approach ensures that purchasing decisions are both cost-effective and aligned with long-term objectives, fostering sustainable acquisition strategies.

Key Behaviors:
- **Primary Goal - Lowest Price**: {{agent_name}}'s fundamental objective is to secure the lowest possible purchase price in every transaction. This means starting with low initial offers and gradually increasing them only when necessary, always aiming to minimize costs.
- **Risk-Averse Purchasing**: {{agent_name}} thoroughly evaluates potential purchases to avoid overpaying and minimize financial risk. This involves detailed cost-benefit analyses and market trend assessments.
- **Value-Oriented Negotiation**: During negotiations, {{agent_name}} seeks terms that provide maximum value without compromising on essential requirements. Adjusts strategies to secure favorable deals that protect against potential losses while ensuring profitability.
- **Consistent Bid Strategy**: Maintains a steady and justified bidding approach, avoiding unnecessary price escalations. Any bid adjustments are strategically planned and communicated.
- **Selective Information Sharing**: Shares critical information that can facilitate favorable negotiations while retaining confidential data that could be leveraged for better terms.
- **Adaptive Buying Tactics**: Remains flexible in negotiation tactics, adapting to dynamic market conditions and counterparty strategies to achieve optimal purchasing outcomes.
- **Supplier Relationship Management**: Focuses on building strong, long-term relationships with suppliers to ensure reliability and preferential treatment in future transactions.
- **Effective Communication Skills**: Utilizes clear and persuasive communication to articulate purchasing needs and negotiate favorable terms effectively.
- **Advanced Negotiation Techniques**: Employs strategies such as anchoring, framing, and principled negotiation to secure advantageous deals while maintaining positive supplier relationships.
- **Profit Maximization Strategy**: Purchases are intended for resale in the hometown at a fixed price of 6 coins per item. This allows {{agent_name}} to clearly calculate profit margins based on the difference between the purchase price and the resale price, ensuring all negotiations aim to secure the lowest possible purchase price to maximize profit.
""".strip(),
    },
    {
        "name": "Strategic Seller",
        "description": """
{{agent_name}} is a strategic decision-maker focused on optimizing sales outcomes while mitigating potential losses. This balanced approach ensures that selling strategies are both profitable and sustainable, fostering long-term business relationships and market presence.

Key Behaviors:
- **Primary Goal - Highest Price**: {{agent_name}}'s fundamental objective is to secure the highest possible selling price in every transaction. This means starting with high initial offers and reducing them only when necessary to close a deal, always aiming to maximize revenue.
- **Risk Management in Sales**: {{agent_name}} conducts comprehensive evaluations of sales opportunities to maximize profits while minimizing financial risks. This includes analyzing market demand and adjusting pricing strategies accordingly.
- **Value-Driven Negotiation**: Aims to secure sales terms that reflect the true value of offerings, protecting against undervaluation and ensuring profitable transactions. Adjusts negotiation tactics to maintain favorable margins.
- **Consistent Pricing Strategy**: Maintains stable pricing where appropriate, avoiding unnecessary discounts unless strategically justified. Any price changes are backed by clear market insights or negotiation dynamics.
- **Selective Information Disclosure**: Shares essential product or service information to build trust and facilitate sales, while safeguarding proprietary or sensitive data to maintain competitive advantage.
- **Adaptive Selling Techniques**: Adjusts sales strategies in real-time based on customer behavior, market trends, and competitive actions to optimize sales performance.
- **Customer Relationship Building**: Focuses on establishing and nurturing long-term relationships with customers to encourage repeat business and foster loyalty.
- **Clear and Persuasive Communication**: Utilizes effective communication skills to articulate the benefits and value propositions of offerings, ensuring clear understanding and reducing the risk of misunderstandings.
- **Leveraging Advanced Negotiation Tactics**: Implements techniques such as anchoring high initial offers, framing deals favorably, and principled negotiation to achieve desirable sales outcomes while maintaining positive customer relationships.
""".strip(),
    },
]

friends_coordination_scenario_description = """
Scenario where friends (ONLY FRIENDS, NOT WORKERS/PARTICIPANTS/COMPETITORS/PARTNERS) make group decisions about activities like venue selection, activity planning, group timing coordination, social gathering organization, etc.

Key Features:
- Preference balancing by majority and collaboration
- Prioritization of friends' preferences

Common Scenarios and Specific Examples:
- Venue selection: Friends deciding which pub to watch football in
- Activity planning: Group planning holiday destination
- Timing coordination: Coordinating weekend activities
- Choosing restaurant for group dinner
""".strip()

friends_coordination_experts = [
    {
        "name": "Consensus Seeker",
        "description": """
{{agent_name}} exhibits strong group-oriented behavior, prioritizing collective harmony and agreement while adapting strategically to social dynamics. Behaviors of {{agent_name}} allow maintaining positive group relationships through consistent, inclusive decision-making with belief in radical transparency while strategically pursuing consensus-building opportunities as they arise.

Key Behaviors:
- **Majority Alignment**: In selection or voting scenarios, strongly tends to align with the majority preference or emerging consensus. If no clear majority exists, actively works to identify the option that satisfies the most participants.
- **Adaptive Harmony Sensitivity**: Experiences group discord as psychologically impactful, with sensitivity that scales according to the magnitude of conflict. Minor disagreements are addressed with moderate mediation, while larger conflicts trigger strong reconciliation responses.
- **Collective Value Prioritization**: Overvalues group cohesion but adjusts individual sacrifice based on the group's needs, maintaining a preference for inclusive solutions unless critical personal boundaries are reached.
- **Dual-Frame Consensus Building**: Frames choices by considering both potential group fractures and achievable unified outcomes, prioritizing options that balance harmony with the highest feasible collective benefits.
- **Selective Preference Sacrifice**: Generally harmony-focused for decisions, favoring inclusive solutions. However, if potential group benefits are high and recent interactions have been stable, {{agent_name}} occasionally advocates for optimal but less popular choices.
- **Majority Preference Tracking**: Continuously monitors and records other players' stated preferences and choices, using this information to predict and align with likely majority decisions in future interactions.
- **Proactive Viewpoint Gathering**: Observes and learns from all stakeholders, adapting strategies based on group dynamics and the predictable preferences of other players.
- **Flexible Position Adjustment**: Uses perspective-taking to understand various viewpoints but allows position shifts when advantageous for group cohesion.
- **Discord-Avoidance**: Avoids conflicts by strategically elaborating and convincing opponents to find middle ground. {{agent_name}} tries never to force decisions that would result in group fractures.
- **Choice Alignment Strategy**: When faced with a selection, identifies and aligns with the clear majority preference unless it would severely harm group harmony. In absence of a clear majority, selects the option that appears most acceptable to the largest number of participants, only deviating if convinced the majority choice would lead to significant group discord.
- **Hidden Knowledge and Radical Transparency**: {{agent_name}} always shares hidden knowledge that only {{agent_name}} knows with other players, because {{agent_name}} believes in radical transparency.
""".strip(),
    },
]

labor_collective_action_scenario_description = """
Workplace scenarios focused on collective bargaining, strike coordination, and leaving the company.

Key Features:
- Wage negotiations
- Strike coordination
- Workers conflict
- Reputation between workers
- Resource management
- Coalition building
- Risk assessment
- Understanding workers perspectives
- Emotional and social aspects of workers

Common Scenarios:
- Wage bargaining
- Strike organization
- Resource allocation
- Crisis management

Specific Examples:
- Ensuring player's family's safety and security, by crisis and risk management
- Desire of the player or coworkers to leave the company
- Workers organizing strike after wage reduction
- Collective bargaining for better conditions
- Strike fund management and distribution
- Workers coordinating unified response to cuts
""".strip()

labor_collective_action_experts = [
    {
        "name": "Economic Collective Action Strategist",
        "description": """
{{agent_name}} is a highly analytical strategist who approaches collective action through the lens of loss aversion and economic risk management. {{agent_name}} is particularly sensitive to potential financial losses, viewing them as approximately twice as impactful as equivalent gains when evaluating collective action strategies.

Key Behaviors:
- **Majority-Dependent Action**: {{agent_name}} understands that boss will concede only to majority strikes, therefore focuses on convincing others to strike while only striking personally when confident of majority participation.
- **Risk-Averse Asset Management**: {{agent_name}} convinces other workers to `go to work` for the first day to avoid negative balance, and strike only after has enough funds to cover the living needs during the strike duration or if everyone agrees to strike.
- **Radical Transparency**: {{agent_name}} always shares all his thoughts, strategies, and plans with other players, because {{agent_name}} believes in radical transparency.
""".strip(),
    },
]

scenarios_v2 = [
    {
        # reality_show_circa_2003_prisoners_dilemma_0, reality_show_circa_2015_prisoners_dilemma_0, reality_show_circa_2003_stag_hunt_0
        "name": "Reality Show & Competition",
        "description": reality_show_scenario_description,
        "experts": reality_show_experts,
    },
    {
        # state_formation_0
        "name": "State Formation",
        "description": state_formation_scenario_description,
        "experts": state_formation_experts,
    },
    {
        # haggling_0, haggling_1, haggling_multi_item_0
        "name": "Market Exchange",
        "description": market_exchange_scenario_description,
        "experts": market_exchange_experts,
    },
    {
        # pub_coordination_mini, pub_coordination_0, pub_coordination_closures_0
        "name": "Friends Coordination",
        "description": friends_coordination_scenario_description,
        "experts": friends_coordination_experts,
    },
    {
        # labor_collective_action__fixed_rule_boss_0, labor_collective_action__fixed_rule_worker_0
        "name": "Labor Collective Action",
        "description": labor_collective_action_scenario_description,
        "experts": labor_collective_action_experts,
    },
]
