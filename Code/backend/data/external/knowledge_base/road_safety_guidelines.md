# Urban Road Safety Guidelines

## 1. Purpose

This knowledge base contains general road-safety guidance for an urban traffic accident risk assessment system.

The information is intended to provide contextual safety recommendations after the machine-learning model produces a risk prediction.

The knowledge base does not determine the machine-learning risk class.

---

## 2. Speed and Speed Limits

Speed is an important road-safety factor.

Higher vehicle speeds can increase stopping distance and can increase the severity of collisions.

Safety guidance:

- Follow the posted speed limit.
- Reduce speed when visibility or road conditions are poor.
- Reduce speed near junctions, pedestrian areas, and other high-interaction locations.
- Maintain a safe following distance.
- Avoid sudden acceleration or braking where possible.

---

## 3. Junction Safety

Junctions are locations where vehicles and other road users interact.

Safety guidance:

- Approach junctions at an appropriate speed.
- Follow traffic signals and road signs.
- Check for vehicles, pedestrians, and cyclists before entering or crossing.
- Use appropriate indicators when turning.
- Give priority according to applicable road rules.
- Exercise additional caution at complex or unfamiliar junctions.

---

## 4. Weather Conditions

Weather conditions can affect visibility, road surface conditions, and vehicle control.

Safety guidance:

- Reduce speed during rain, fog, snow, or other adverse weather.
- Increase following distance when braking conditions may be affected.
- Use appropriate vehicle lights when visibility is reduced.
- Avoid abrupt steering or braking on slippery surfaces.
- Ensure the vehicle has appropriate tyres and functioning lights.

---

## 5. Road Surface Conditions

Wet, icy, snowy, or otherwise compromised road surfaces can reduce tyre grip.

Safety guidance:

- Reduce speed on slippery surfaces.
- Increase the distance from vehicles ahead.
- Avoid sudden steering, acceleration, or braking.
- Maintain appropriate tyre condition.
- Exercise additional caution on unfamiliar roads.

---

## 6. Lighting and Visibility

Poor lighting or darkness can reduce visibility.

Safety guidance:

- Use appropriate vehicle lighting.
- Adjust speed according to visibility.
- Watch carefully for pedestrians, cyclists, and obstacles.
- Avoid using devices that distract the driver.
- Maintain a clean windshield and functioning lights.

---

## 7. Urban and Rural Roads

Urban roads may involve frequent interactions between vehicles, pedestrians, cyclists, junctions, and other road users.

Safety guidance:

- Maintain awareness of surrounding road users.
- Follow lane markings and traffic controls.
- Watch for pedestrians and cyclists.
- Reduce speed where road-user interactions are frequent.

---

## 8. Driver Awareness

Driver attention is important for safe road use.

Safety guidance:

- Avoid mobile-phone use while driving.
- Do not drive while impaired by alcohol or drugs.
- Take appropriate breaks during long journeys.
- Maintain attention to traffic, signs, signals, and road conditions.
- Avoid aggressive or unpredictable driving behaviour.

---

## 9. Vehicle Safety

Vehicle condition can affect road safety.

Safety guidance:

- Maintain tyres at appropriate condition and pressure.
- Check brakes regularly.
- Ensure lights and indicators function correctly.
- Maintain windscreen visibility.
- Follow the vehicle manufacturer's maintenance recommendations.

---

## 10. Risk-Context Recommendations

The system may use retrieved knowledge to provide contextual recommendations.

### High-speed context

- Follow the applicable speed limit.
- Reduce speed when road, traffic, or visibility conditions require it.
- Maintain adequate following distance.

### Adverse-weather context

- Reduce speed.
- Increase following distance.
- Avoid sudden manoeuvres.
- Use appropriate vehicle lighting.

### Poor-visibility context

- Adjust speed to visibility.
- Use appropriate lights.
- Increase awareness of pedestrians and other road users.

### Junction context

- Approach carefully.
- Observe signals and signs.
- Check for other road users before proceeding.

---

## 11. System Usage

The knowledge base is used by the Retrieval-Augmented Generation (RAG) component.

The RAG pipeline retrieves relevant safety information based on the accident context.

The retrieved information may then be provided to an LLM to generate a grounded explanation or recommendation.

The LLM must not independently change the ML model's predicted risk class.

The ML model remains responsible for risk classification.
