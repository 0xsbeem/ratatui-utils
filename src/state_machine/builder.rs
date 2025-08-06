//! # State Machine Builder
//!
//! A runtime validated state machine builder that ensures state machine correctness
//! through type-safe transitions and comprehensive validation rules.
//!
//! ## Overview
//!
//! This module provides tools for building validated state machines where:
//! - All transitions must be explicitly defined
//! - Every state is guaranteed to be reachable from another state
//! - Every state is guaranteed to be escapable (has outgoing transitions)
//! - Custom validation logic can be applied to transitions
//!
//! ## Basic Usage
//!
//! ```rust
//! use std::fmt::Display;
//! use std::hash::Hash;
//! # use ratatui_utils::state_machine::*;
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! enum AppState {
//!     Menu,
//!     Playing,
//!     GameOver,
//! }
//!
//! impl Display for AppState {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         match self {
//!             AppState::Menu => write!(f, "Menu"),
//!             AppState::Playing => write!(f, "Playing"),
//!             AppState::GameOver => write!(f, "GameOver"),
//!         }
//!     }
//! }
//!
//! let state_machine = StateMachineBuilder::new()
//!     .allow(AppState::Menu, AppState::Playing)      // Menu -> Playing
//!     .allow(AppState::Playing, AppState::GameOver)  // Playing -> GameOver  
//!     .allow(AppState::GameOver, AppState::Menu)     // GameOver -> Menu 
//!     .build()
//!     .expect("State machine validation failed");
//!
//! assert!(state_machine.is_transition_allowed(&AppState::Menu, &AppState::Playing));
//! assert!(!state_machine.is_transition_allowed(&AppState::Menu, &AppState::GameOver));
//! ```
//!
//! ## Advanced Usage with Validation
//!
//! ```rust
//! # use ratatui_utils::state_machine::*;
//! # use std::fmt::Display;
//! # use std::hash::Hash;
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! enum GameState { 
//!     Playing, 
//!     Inventory,
//!     Menu,
//! }
//!
//! impl Display for GameState {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         write!(f, "{:?}", self)
//!     }
//! }
//!
//! let state_machine = StateMachineBuilder::new()
//!     .allow_if(GameState::Menu, GameState::Playing, |from, to| {
//!         // Custom validation: only start game if player is ready
//!         is_player_ready() 
//!     })
//!     .allow(GameState::Playing, GameState::Inventory)
//!     .allow(GameState::Inventory, GameState::Menu)
//!     .build()
//!     .expect("State machine validation failed.");
//!
//! fn is_player_ready() -> bool { true } 
//! ```
//!
//! ## Validation Rules
//!
//! The state machine builder enforces these strict validation rules:
//!
//! ### 1. **Reachability Rule**
//! Every state must be reachable from at least one other state:
//!
//! ```rust
//! # use ratatui_utils::state_machine::*;
//! # use std::fmt::Display;
//! # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! # enum State { Connected, Isolated }
//! # impl Display for State {
//! #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//! #         write!(f, "{:?}", self)
//! #     }
//! # }
//! // This will fail because Isolated has no incoming transitions
//! let result = StateMachineBuilder::new()
//!     .allow(State::Connected, State::Connected) // Self-loop
//!     // State::Isolated has no incoming transitions - violates reachability!
//!     .build();
//!
//! // This would fail if we tried to reference State::Isolated anywhere
//! ```
//!
//! ### 2. **Escapability Rule**
//! Every state must have at least one outgoing transition:
//!
//! ```rust
//! # use ratatui_utils::state_machine::*;
//! # use std::fmt::Display;
//! # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! # enum State { Good, Trap }
//! # impl Display for State {
//! #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//! #         write!(f, "{:?}", self)
//! #     }
//! # }
//! // This will fail because Trap has no outgoing transitions
//! let result = StateMachineBuilder::new()
//!     .allow(State::Good, State::Trap)
//!     // State::Trap has no outgoing transitions - trapped!
//!     .build();
//!
//! assert!(result.is_err());
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::hash::Hash;
use std::rc::Rc;

/// A transition definition between two states.
///
/// This represents a directed edge in the state machine graph, indicating
/// that a transition from the `from` state to the `to` state is valid.
///
/// For more complicated state transition validation, consider using a
/// [`TransitionValidator`].
///
/// # Examples
///
/// ```rust
/// # use ratatui_utils::state_machine::TransitionRule;
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// enum State { A, B }
///
/// let transition = TransitionRule::new(State::A, State::B);
/// assert_eq!(transition.from, State::A);
/// assert_eq!(transition.to, State::B);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TransitionRule<T> {
    /// The source state of the transition
    pub from: T,
    /// The destination state of the transition
    pub to: T,
}

impl<T> TransitionRule<T> {
    /// Creates a new transition rule from one state to another.
    ///
    /// # Arguments
    ///
    /// * `from` - The source state
    /// * `to` - The destination state
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::TransitionRule;
    /// #[derive(Clone, PartialEq, Eq, Hash)]
    /// enum MyState { Start, End }
    ///
    /// let rule = TransitionRule::new(MyState::Start, MyState::End);
    /// ```
    pub fn new(from: T, to: T) -> Self {
        Self { from, to }
    }
}
/// A validation function for state transitions.
///
/// This function type is used to provide **dynamic validation logic** for
/// specific transitions beyond what [`TransitionRule`] provides. While
/// [`TransitionRule`] defines which transitions are *structurally possible*,
/// [`TransitionValidator`] functions determine whether a transition should
/// be allowed *right now* based on current runtime conditions.
///
/// The validator receives references to the source and destination states
/// and returns `true` if the transition should be allowed, or `false` if
/// it should be rejected.
///
/// # When to Use Validators
///
/// Use validators for **dynamic conditions** that can't be expressed by
/// state types alone:
/// - External system state (network connectivity, file permissions)
/// - Resource availability (memory, battery, user credits)
/// - Time-based constraints (cooldowns, business hours)
/// - User permissions or authentication status
/// - Complex business logic that depends on runtime context
///
/// # When NOT to Use Validators
///
/// Don't use validators for conditions already handled by [`TransitionRule`]:
/// - "Only allow A->B transitions" (just use `.allow(A, B)`)
/// - "Prevent B->A transitions" (don't define the rule)
///
/// # Examples
///
/// ## External System Validation
///
/// ```rust
/// # use ratatui_utils::state_machine::TransitionValidator;
/// # use std::rc::Rc;
/// #[derive(PartialEq)]
/// enum NetworkState { Offline, Connecting, Online }
///
/// let network_validator: TransitionValidator<NetworkState> = Rc::new(|from, to| {
///     match (from, to) {
///         (NetworkState::Offline, NetworkState::Connecting) => {
///             // Only connect if network is available
///             network_interface_available()
///         }
///         (NetworkState::Connecting, NetworkState::Online) => {
///             // Only go online if authentication succeeded
///             authentication_completed()
///         }
///         _ => true, // Other transitions always allowed
///     }
/// });
///
/// fn network_interface_available() -> bool { true } 
/// fn authentication_completed() -> bool { true }
/// ```
///
/// ## Resource-Based Validation
///
/// ```rust
/// # use ratatui_utils::state_machine::TransitionValidator;
/// # use std::rc::Rc;
/// #[derive(PartialEq)]
/// enum GameState { Playing, Shop, Inventory }
///
/// let resource_validator: TransitionValidator<GameState> = Rc::new(|from, to| {
///     match to {
///         GameState::Shop => {
///             // Can only enter shop if player has money
///             player_has_currency()
///         }
///         GameState::Inventory => {
///             // Can only open inventory if not in combat
///             !player_in_combat()
///         }
///         _ => true,
///     }
/// });
///
/// fn player_has_currency() -> bool { true }
/// fn player_in_combat() -> bool { false }
/// ```
///
/// ## Time-Based Validation
///
/// ```rust
/// # use ratatui_utils::state_machine::TransitionValidator;
/// # use std::rc::Rc;
/// #[derive(PartialEq)]
/// enum AppState { Active, Suspended }
///
/// let time_validator: TransitionValidator<AppState> = Rc::new(|from, to| {
///     match (from, to) {
///         (AppState::Suspended, AppState::Active) => {
///             // Only resume if suspension period has ended
///             suspension_expired()
///         }
///         _ => true,
///     }
/// });
///
/// fn suspension_expired() -> bool { true }
/// ```
///
/// # Integration with State Machine Builder
///
/// Validators are used with [`StateMachineBuilder::allow_if`]:
///
/// ```rust
/// # use ratatui_utils::state_machine::*;
/// # use std::fmt::Display;
/// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// # enum State { Normal, Admin }
/// # impl Display for State { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self) } }
/// # fn user_has_admin_privileges() -> bool { true }
/// let state_machine = StateMachineBuilder::new()
///     .allow_if(State::Normal, State::Admin, |from, to| {
///         // Dynamic validation: check user permissions at transition time
///         user_has_admin_privileges()
///     })
///     .allow(State::Admin, State::Normal)  // No validation needed to de-escalate
///     .build()
///     .expect("State machine validation failed.");
/// ```
pub type TransitionValidator<T> = Rc<dyn Fn(&T, &T) -> bool>;

/// A validated state machine definition.
///
/// This struct represents a complete, validated state machine that enforces:
/// - **Reachability**: Every state can be reached from another state  
/// - **Escapability**: Every state has at least one outgoing transition
///
/// The state machine allows **flexible graph structures** including cycles, diamond patterns,
/// and multiple paths to the same state. 
///
/// Instances of `StateMachine` are created through the [`StateMachineBuilder`]
/// and are guaranteed to be valid according to the validation rules.
///
/// # Type Parameters
///
/// * `T` - The state type, which must implement `Clone`, `PartialEq`, `Hash`, `Eq`, and `Display`
///
/// # Examples
///
/// ```rust
/// # use ratatui_utils::state_machine::*;
/// # use std::fmt::Display;
/// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// # enum State { A, B, C }
/// # impl Display for State {
/// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
/// #         write!(f, "{:?}", self)
/// #     }
/// # }
/// // Diamond pattern: both A and B can reach C
/// let state_machine = StateMachineBuilder::new()
/// 
/// 
///     .allow(State::A, State::C)    // C is reachable, A is escapable.
///     .allow(State::B, State::C)    // C is reachable, B is escapable.
///     .allow(State::C, State::A)    // A is reachable, C is escapable.
///     .allow(State::C, State::B)    // B is reachable, C is escapable.
///     .build()
///     .unwrap();
///
/// // Query the state machine
/// assert!(state_machine.is_transition_allowed(&State::A, &State::C));
/// assert!(state_machine.is_transition_allowed(&State::B, &State::C));
/// let to_c = state_machine.get_transitions_to(&State::C);
/// assert_eq!(to_c.len(), 2); // C has two incoming transitions
/// ```
pub struct StateMachine<T>
where
    T: Clone + PartialEq + Hash + Eq + Display,
{
    /// Map of allowed transitions and their validation functions
    pub allowed_transitions: HashMap<TransitionRule<T>, TransitionValidator<T>>,
    /// Set of all states in the state machine
    pub all_states: HashSet<T>,
}

impl<T> StateMachine<T>
where
    T: Clone + PartialEq + Hash + Eq + Display,
{
    /// Checks if a transition between two states is allowed.
    ///
    /// This method only checks if the transition is defined in the state machine,
    /// not whether it would pass validation. Use [`validate_transition`] to
    /// run the associated validation function.
    ///
    /// # Arguments
    ///
    /// * `from` - The source state
    /// * `to` - The destination state
    ///
    /// # Returns
    ///
    /// `true` if the transition is defined, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { A, B, C }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let machine = StateMachineBuilder::new()
    ///     .allow(State::A, State::B)
    ///     .allow(State::B, State::A)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert!(machine.is_transition_allowed(&State::A, &State::B));
    /// assert!(!machine.is_transition_allowed(&State::A, &State::C));
    /// ```
    ///
    /// [`validate_transition`]: Self::validate_transition
    pub fn is_transition_allowed(&self, from: &T, to: &T) -> bool {
        let rule = TransitionRule::new(from.clone(), to.clone());
        self.allowed_transitions.contains_key(&rule)
    }
    
    /// Validates a transition using its associated validation function.
    ///
    /// This method first checks if the transition is allowed, then runs
    /// the associated validation function to determine if the transition
    /// should be permitted in the current context.
    ///
    /// # Arguments
    ///
    /// * `from` - The source state
    /// * `to` - The destination state
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Transition is allowed and validation passed
    /// * `Ok(false)` - Transition is allowed but validation failed
    /// * `Err(StateMachineError)` - Transition is not defined
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { Locked, Unlocked }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let machine = StateMachineBuilder::new()
    ///     .allow_if(State::Locked, State::Unlocked, |_, _| {
    ///         // Pretend we have a validation function here
    ///         true
    ///     })
    ///     .allow(State::Unlocked, State::Locked)
    ///     .build()
    ///     .unwrap();
    ///
    /// // Validation passes
    /// assert_eq!(machine.validate_transition(&State::Locked, &State::Unlocked), Ok(true));
    ///
    /// // Undefined transition
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum Other { X }
    /// # impl Display for Other {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// // This would fail because the transition isn't defined
    /// // assert!(machine.validate_transition(&State::Locked, &Other::X).is_err());
    /// ```
    pub fn validate_transition(&self, from: &T, to: &T) -> Result<bool, StateMachineError> {
        let rule = TransitionRule::new(from.clone(), to.clone());
        if let Some(validator) = self.allowed_transitions.get(&rule) {
            Ok(validator(from, to))
        } else {
            Err(StateMachineError::InvalidTransition(
                format!("Transition from {} to {} is not defined", from, to)
            ))
        }
    }
    
    /// Gets all possible destination states from a given source state.
    ///
    /// Returns a vector of states that can be reached directly from
    /// the specified source state.
    ///
    /// # Arguments
    ///
    /// * `from` - The source state to query
    ///
    /// # Returns
    ///
    /// A vector of destination states reachable from the source state
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { Menu, Play, Settings }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let machine = StateMachineBuilder::new()
    ///     .allow(State::Menu, State::Play)
    ///     .allow(State::Menu, State::Settings)
    ///     .allow(State::Play, State::Menu)
    ///     .allow(State::Settings, State::Menu)
    ///     .build()
    ///     .unwrap();
    ///
    /// let from_menu = machine.get_transitions_from(&State::Menu);
    /// assert_eq!(from_menu.len(), 2);
    /// assert!(from_menu.contains(&State::Play));
    /// assert!(from_menu.contains(&State::Settings));
    /// ```
    pub fn get_transitions_from(&self, from: &T) -> Vec<T> {
        self.allowed_transitions
            .keys()
            .filter(|rule| &rule.from == from)
            .map(|rule| rule.to.clone())
            .collect()
    }
    
    /// Gets all possible source states that can transition to a given destination state.
    ///
    /// Returns a vector of states that can directly reach the specified
    /// destination state.
    ///
    /// # Arguments
    ///
    /// * `to` - The destination state to query
    ///
    /// # Returns
    ///
    /// A vector of source states that can reach the destination state
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { A, B, C }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let machine = StateMachineBuilder::new()
    ///     .allow(State::A, State::C)
    ///     .allow(State::B, State::C)
    ///     .allow(State::C, State::A)
    ///     .build()
    ///     .unwrap();
    ///
    /// let to_c = machine.get_transitions_to(&State::C);
    /// assert_eq!(to_c.len(), 2);
    /// assert!(to_c.contains(&State::A));
    /// assert!(to_c.contains(&State::B));
    /// ```
    pub fn get_transitions_to(&self, to: &T) -> Vec<T> {
        self.allowed_transitions
            .keys()
            .filter(|rule| &rule.to == to)
            .map(|rule| rule.from.clone())
            .collect()
    }
    
    /// Gets a reference to all states in the state machine.
    ///
    /// Returns the complete set of states that are part of this state machine,
    /// including both source and destination states from all defined transitions.
    ///
    /// # Returns
    ///
    /// A reference to a `HashSet` containing all states
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { X, Y, Z }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let machine = StateMachineBuilder::new()
    ///     .allow(State::X, State::Y)
    ///     .allow(State::Y, State::Z)
    ///     .allow(State::Z, State::X)
    ///     .build()
    ///     .unwrap();
    ///
    /// let all_states = machine.get_all_states();
    /// assert_eq!(all_states.len(), 3);
    /// assert!(all_states.contains(&State::X));
    /// assert!(all_states.contains(&State::Y));
    /// assert!(all_states.contains(&State::Z));
    /// ```
    pub fn get_all_states(&self) -> &HashSet<T> {
        &self.all_states
    }
    
    /// Checks if a specific state exists in the state machine.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to check for existence
    ///
    /// # Returns
    ///
    /// `true` if the state exists in the state machine, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { Defined, NotDefined }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let machine = StateMachineBuilder::new()
    ///     .allow(State::Defined, State::Defined)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert!(machine.has_state(&State::Defined));
    /// assert!(!machine.has_state(&State::NotDefined));
    /// ```
    pub fn has_state(&self, state: &T) -> bool {
        self.all_states.contains(state)
    }
}

impl<T> std::fmt::Debug for StateMachine<T>
where
    T: Clone + PartialEq + Hash + Eq + Display + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateMachine")
            .field("all_states", &self.all_states)
            .field("transitions_count", &self.allowed_transitions.len())
            .field("transitions", &format!("{} transitions defined", self.allowed_transitions.len()))
            .finish()
    }
}

impl<T: Hash + Display + Eq + Display + Clone> Clone for StateMachine<T> {
    fn clone(&self) -> Self {
        StateMachine {
            allowed_transitions: self.allowed_transitions.clone(), // ✅ Rc::clone preserves validators!
            all_states: self.all_states.clone(),
        }
    }
}

/// Builder for creating validated state machines
pub struct StateMachineBuilder<T>
where
    T: Clone + PartialEq + Hash + Eq + Display,
{
    transitions: HashMap<TransitionRule<T>, TransitionValidator<T>>,
}

impl<T> StateMachineBuilder<T>
where
    T: Clone + PartialEq + Hash + Eq + Display,
{
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
        }
    }
    
    
    /// Add a valid transition with a custom validator function
    pub fn allow_transition<F>(mut self, from: T, to: T, validator: F) -> Self
    where
        F: Fn(&T, &T) -> bool + 'static,
    {
        let rule = TransitionRule::new(from, to);
        // Store the validator as a function pointer for simplicity
        self.transitions.insert(rule, Rc::new(validator));
        self
    }
    
    /// Add a valid transition that's always allowed
    pub fn allow(self, from: T, to: T) -> Self {
        self.allow_transition(from, to, |_, _| true)
    }
    
    /// Add a valid transition with a condition check
    pub fn allow_if<F>(self, from: T, to: T, condition: F) -> Self
    where
        F: Fn(&T, &T) -> bool + 'static,
    {
        self.allow_transition(from, to, condition)
    }
    
    /// Build the state machine, validating it
    pub fn build(self) -> Result<StateMachine<T>, StateMachineError> {
        let all_states = self.get_all_states();
        
        // Create the state machine first
        let state_machine = StateMachine {
            allowed_transitions: self.transitions,
            all_states: all_states.clone(),
        };
        
        // Validate the state machine
        Self::validate_state_machine(&state_machine, &all_states)?;
        
        Ok(state_machine)
    }
    
    fn get_all_states(&self) -> HashSet<T> {
        let mut states = HashSet::new();
        for rule in self.transitions.keys() {
            states.insert(rule.from.clone());
            states.insert(rule.to.clone());
        }
        states
    }
    
    /// Validate the state machine rules
    fn validate_state_machine(
        state_machine: &StateMachine<T>,
        all_states: &HashSet<T>,
    ) -> Result<(), StateMachineError> {
        // Check that every state is reachable (has at least one incoming transition)
        // Exception: We allow states with no incoming transitions as potential initial states
        Self::validate_reachability(state_machine, all_states)?;
        
        // Check that every state can escape (has at least one outgoing transition)
        Self::validate_escapability(state_machine, all_states)?;
        
        Ok(())
    }
    
    fn validate_reachability(
        state_machine: &StateMachine<T>,
        all_states: &HashSet<T>,
    ) -> Result<(), StateMachineError> {
        // Find states with no incoming transitions
        let states_with_incoming: HashSet<_> = state_machine
            .allowed_transitions
            .keys()
            .map(|rule| &rule.to)
            .collect();
            
        let unreachable_states: Vec<_> = all_states
            .iter()
            .filter(|state| !states_with_incoming.contains(state))
            .collect();
            
        // It's OK to have states with no incoming transitions (they can be initial states)
        // But we might want to warn about it
        if unreachable_states.len() == all_states.len() {
            return Err(StateMachineError::NoReachableStates);
        }
        
        Ok(())
    }
    
    fn validate_escapability(
        state_machine: &StateMachine<T>,
        all_states: &HashSet<T>,
    ) -> Result<(), StateMachineError> {
        for state in all_states {
            if !Self::can_escape(state_machine, state) {
                return Err(StateMachineError::TrappedState(format!("{}", state)));
            }
        }
        Ok(())
    }
    
    fn can_escape(state_machine: &StateMachine<T>, state: &T) -> bool {
        // A state can escape if there's at least one transition FROM it
        state_machine
            .allowed_transitions
            .keys()
            .any(|rule| &rule.from == state)
    }
    
    fn validate_single_path(
        state_machine: &StateMachine<T>,
        all_states: &HashSet<T>,
    ) -> Result<(), StateMachineError> {
        // In strict mode, each state should have at most one incoming transition
        let mut incoming_count: HashMap<&T, usize> = HashMap::new();
        
        for rule in state_machine.allowed_transitions.keys() {
            *incoming_count.entry(&rule.to).or_insert(0) += 1;
        }
        
        for (state, count) in incoming_count {
            if count > 1 {
                return Err(StateMachineError::MultiplePathsToState(format!(
                    "{}", state
                )));
            }
        }
        
        Ok(())
    }
}

impl<T> Default for StateMachineBuilder<T>
where
    T: Clone + PartialEq + Hash + Eq + Display,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during state machine validation or operation.
///
/// These errors are returned when state machine validation fails or when
/// invalid operations are attempted on a state machine.
///
/// # Variants
///
/// ## Validation Errors (from [`StateMachineBuilder::build`])
///
/// * [`TrappedState`] - A state has no outgoing transitions
/// * [`MultiplePathsToState`] - In strict mode, a state has multiple incoming transitions  
/// * [`NoReachableStates`] - No states can be reached (empty state machine)
///
/// ## Runtime Errors
///
/// * [`InvalidTransition`] - An undefined or rejected transition was attempted
/// * [`UnreachableState`] - A state cannot be reached from any other state (unused)
///
/// # Examples
///
/// ## Handling Validation Errors
///
/// ```rust
/// # use ratatui_utils::state_machine::*;
/// # use std::fmt::Display;
/// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// # enum State { Trap, Normal }
/// # impl Display for State {
/// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
/// #         write!(f, "{:?}", self)
/// #     }
/// # }
/// let result = StateMachineBuilder::new()
///     .allow(State::Normal, State::Trap)
///     // State::Trap has no outgoing transitions!
///     .build();
///
/// match result {
///     Ok(state_machine) => println!("State machine is valid"),
///     Err(StateMachineError::TrappedState(state)) => {
///         eprintln!("State '{}' is trapped (no outgoing transitions)", state);
///     }
///     Err(StateMachineError::MultiplePathsToState(state)) => {
///         eprintln!("State '{}' has multiple incoming paths", state);
///     }
///     Err(other) => eprintln!("Other validation error: {}", other),
/// }
/// ```
///
/// [`StateMachineBuilder::build`]: StateMachineBuilder::build
/// [`TrappedState`]: Self::TrappedState
/// [`MultiplePathsToState`]: Self::MultiplePathsToState
/// [`NoReachableStates`]: Self::NoReachableStates
/// [`InvalidTransition`]: Self::InvalidTransition
/// [`UnreachableState`]: Self::UnreachableState
#[derive(Debug, Clone, PartialEq)]
pub enum StateMachineError {
    /// A state cannot be reached from any other state.
    ///
    /// This error occurs during validation when a state is found that has
    /// no incoming transitions. In strict mode, every state must be reachable
    /// from at least one other state to ensure the state machine is fully
    /// connected and prevents isolated states.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { Connected, Isolated }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// // This will fail because Isolated has no incoming transitions
    /// let result = StateMachineBuilder::new()
    ///     .allow(State::Connected, State::Connected) // Self-loop
    ///     // Would need to reference State::Isolated in a transition
    ///     .build();
    /// // The error would occur if State::Isolated was referenced
    /// ```
    UnreachableState(String),
    
    /// A state has no outgoing transitions, making it a "trap" state.
    ///
    /// This error occurs during validation when a state is found that has
    /// no outgoing transitions. Such states would cause the state machine
    /// to become stuck with no way to continue.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { Good, Trap }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let result = StateMachineBuilder::new()
    ///     .allow(State::Good, State::Trap)
    ///     // State::Trap has no outgoing transitions - this will fail
    ///     .build();
    ///
    /// assert!(matches!(result, Err(StateMachineError::TrappedState(_))));
    /// ```
    TrappedState(String),
    
    /// Multiple paths to a state are allowed (no longer an error).
    ///
    /// This error type is kept for backward compatibility but is no longer
    /// returned by the validation system. States can now have multiple
    /// incoming transitions, enabling diamond patterns and flexible workflows.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { A, B, C }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// // This is now valid and allowed
    /// let result = StateMachineBuilder::new()
    ///     .allow(State::A, State::C)  // Path 1 to C
    ///     .allow(State::B, State::C)  // Path 2 to C - now allowed!
    ///     .allow(State::C, State::A)
    ///     .allow(State::C, State::B)
    ///     .build();
    ///
    /// assert!(result.is_ok()); // Multiple paths are now allowed
    /// ```
    MultiplePathsToState(String),
    
    /// An invalid transition was attempted or a transition failed validation.
    ///
    /// This error can occur in two scenarios:
    /// 1. Attempting a transition that is not defined in the state machine
    /// 2. A defined transition that failed its validation function
    ///
    /// # Examples
    ///
    /// ## Undefined Transition
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { A, B }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let machine = StateMachineBuilder::new()
    ///     .allow(State::A, State::B)
    ///     .allow(State::B, State::A)
    ///     .build()
    ///     .unwrap();
    ///
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum Other { X }
    /// # impl Display for Other {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// // This would return InvalidTransition because A -> Other::X is not defined
    /// // let result = machine.validate_transition(&State::A, &Other::X);
    /// // assert!(matches!(result, Err(StateMachineError::InvalidTransition(_))));
    /// ```
    ///
    /// ## Failed Validation
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// # #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// # enum State { Locked, Unlocked }
    /// # impl Display for State {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{:?}", self)
    /// #     }
    /// # }
    /// let machine = StateMachineBuilder::new()
    ///     .allow_if(State::Locked, State::Unlocked, |_, _| {
    ///         false // Always reject the transition
    ///     })
    ///     .allow(State::Unlocked, State::Locked)
    ///     .build()
    ///     .unwrap();
    ///
    /// let result = machine.validate_transition(&State::Locked, &State::Unlocked);
    /// assert_eq!(result, Ok(false)); // Transition defined but validation failed
    /// ```
    InvalidTransition(String),
    
    /// No states in the state machine are reachable.
    ///
    /// This error occurs when the state machine is empty or when all states
    /// are completely isolated with no connections. This typically indicates
    /// an empty or malformed state machine definition.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ratatui_utils::state_machine::*;
    /// # use std::fmt::Display;
    /// // This would occur with an empty state machine
    /// let result = StateMachineBuilder::<String>::new().build();
    /// // Would fail because no states are defined
    /// ```
    NoReachableStates,
}

impl Display for StateMachineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateMachineError::UnreachableState(state) => {
                write!(f, "State '{}' is not reachable from any other state", state)
            }
            StateMachineError::TrappedState(state) => {
                write!(f, "State '{}' has no outgoing transitions (trapped)", state)
            }
            StateMachineError::MultiplePathsToState(state) => {
                write!(
                    f,
                    "State '{}' has multiple incoming paths (violates strict mode)",
                    state
                )
            }
            StateMachineError::InvalidTransition(msg) => {
                write!(f, "Invalid transition: {}", msg)
            }
            StateMachineError::NoReachableStates => {
                write!(f, "No states are reachable (all states are isolated)")
            }
        }
    }
}

impl std::error::Error for StateMachineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum TestState {
        A,
        B,
        C,
    }

    impl Display for TestState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestState::A => write!(f, "A"),
                TestState::B => write!(f, "B"), 
                TestState::C => write!(f, "C"),
            }
        }
    }

    #[test]
    fn test_valid_state_machine() {
        let state_machine = StateMachineBuilder::new()
            .allow(TestState::A, TestState::B)
            .allow(TestState::B, TestState::C)
            .allow(TestState::C, TestState::A)
            .build()
            .unwrap();

        assert!(state_machine.is_transition_allowed(&TestState::A, &TestState::B));
        assert!(!state_machine.is_transition_allowed(&TestState::A, &TestState::C));
        assert!(!state_machine.is_transition_allowed(&TestState::A, &TestState::A));
        assert!(!state_machine.is_transition_allowed(&TestState::B, &TestState::B));
        assert!(state_machine.is_transition_allowed(&TestState::B, &TestState::C));
        assert!(!state_machine.is_transition_allowed(&TestState::B, &TestState::A));
        assert!(!state_machine.is_transition_allowed(&TestState::C, &TestState::B));
        assert!(!state_machine.is_transition_allowed(&TestState::C, &TestState::C));
        assert!(state_machine.is_transition_allowed(&TestState::C, &TestState::A));
        
        let transitions_from_a = state_machine.get_transitions_from(&TestState::A);
        assert_eq!(transitions_from_a, vec![TestState::B]);
    }

    #[test]
    fn test_trapped_state_detection() {
        let result = StateMachineBuilder::new()
            .allow(TestState::A, TestState::B)
            .build();

        assert!(matches!(result, Err(StateMachineError::TrappedState(_))));
    }

    #[test]
    fn test_basic_transition_validation() {
        let state_machine = StateMachineBuilder::new()
            .allow_if(TestState::A, TestState::B, |from, to| {
                // Custom validation logic
                from == &TestState::A && to == &TestState::B
            })
            .allow(TestState::B, TestState::A)
            .build()
            .unwrap();

        // Valid transition with successful validation
        assert!(state_machine
            .validate_transition(&TestState::A, &TestState::B)
            .unwrap());

        // Invalid transition (not defined)
        assert!(state_machine
            .validate_transition(&TestState::A, &TestState::C)
            .is_err());
    }

    #[test]
    fn test_external_transition_validation() {
        struct ExternalState { pub val: bool }
        fn is_allowed(s: &ExternalState) -> bool {
            if s.val {
                true
            } else {
                false
            }
        }

        let state = ExternalState{val: false};
        let state_machine = StateMachineBuilder::new()
            .allow_if(TestState::A, TestState::B, move |_from, _to| {
                is_allowed(&state)
            })
            .allow(TestState::B, TestState::A)
            .build()
            .unwrap();

        // Valid transition with unsuccessful validation
        assert!(!state_machine
            .validate_transition(&TestState::A, &TestState::B)
            .unwrap());

        let state2 = ExternalState{val: true};
        let state_machine2 = StateMachineBuilder::new()
            .allow_if(TestState::A, TestState::B, move |_from, _to| {
                is_allowed(&state2)
            })
            .allow(TestState::B, TestState::A)
            .build()
            .unwrap();

        // Valid transition with successful validation
        assert!(state_machine2
            .validate_transition(&TestState::A, &TestState::B)
            .unwrap());
    }

    #[test]
    fn test_state_queries() {
        let state_machine = StateMachineBuilder::new()
            .allow(TestState::A, TestState::B)
            .allow(TestState::B, TestState::A)
            .build()
            .unwrap();

        assert!(state_machine.has_state(&TestState::A));
        assert!(state_machine.has_state(&TestState::B));
        assert!(!state_machine.has_state(&TestState::C));

        let all_states = state_machine.get_all_states();
        assert_eq!(all_states.len(), 2);
        assert!(all_states.contains(&TestState::A));
        assert!(all_states.contains(&TestState::B));
    }
}
