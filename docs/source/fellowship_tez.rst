.. _festim-fellowship-tez:

FESTIM Fellowship — Tez
========================

.. rubric:: UKAEA · First FESTIM Fellow · 2025

.. rst-class:: lead

   **Tez** is a physicist and materials scientist at :bdg-primary:`UKAEA` — and our
   **very first FESTIM Fellow**. She joined the FESTIM development team at MIT for
   two months in 2025 to master FESTIM v2 and bring that expertise back to UKAEA.

.. grid:: 2
   :gutter: 4

   .. grid-item::
      :columns: 12 5 5 5

      .. figure:: /_static/images/fellows/tez_portrait.jpg
         :alt: Tez, FESTIM Fellow, UKAEA
         :align: center
         :width: 100%

         *Tez at MIT during her fellowship.*

      .. TODO: replace with actual photo asset.

   .. grid-item::
      :columns: 12 7 7 7

      .. grid:: 1
         :gutter: 2

         .. grid-item::

            .. card::
               :class-card: sd-border-1

               🗓️ **2 months at MIT** — in person

         .. grid-item::

            .. card::
               :class-card: sd-border-1

               🐛 **1 bug fixed** — contributed directly to FESTIM v2

         .. grid-item::

            .. card::
               :class-card: sd-border-1

               🔬 **LIBRTI + STEP** — UKAEA programmes now using FESTIM

----

Background & Motivation
-----------------------

.. rubric:: 1. What's your background, and what drew you to apply for the FESTIM Fellowship?

I have a physics and materials science background. I have worked on modelling of
**tritium transport** in **plasma facing components** and **lithium-based tritium breeders**
in the past using FESTIM and COMSOL. I was interested to try out the FESTIM fellowship
as FESTIM v1 was the first modelling code I worked with and learning to use it for the
cases I was looking at was quite easy. The FESTIM fellowship meant that I could learn
the new capabilities of **FESTIM v2** to help with more advanced tritium transport
modelling in breeders at UKAEA while developing my own modelling skillset.

.. rubric:: 2. What was your experience with FESTIM before starting the fellowship?

I had used **FESTIM v1.4** when I started working at UKAEA to model tritium transport
in STEP-like plasma facing components. I began using FESTIM because it could implement
**tritium trapping** and **boundary conditions** easily. I also used FESTIM v1.4 to
make a faux model of **multi-level trapping** in a material.

.. rubric:: 3. What specific project did you bring to the fellowship, and why was it important to your work at UKAEA?

I had hoped to learn FESTIM v2 capabilities and work closely with the FESTIM team as
part of my development, so I only brought the plasma facing component, specifically a
**monoblock**, case to the fellowship to learn based off something I knew the results
of already. This meant that I could **push the limits of the model**, exploring edge
cases, and know what to expect out of the model.

----

The Experience
--------------

.. rubric:: 4. How would you describe working with the FESTIM development team? Were there any standout moments or interactions?

I had a wonderful time working with the FESTIM development team, they were very easy
to talk to and would answer any questions I had with enthusiasm. It was really helpful
to have them so close by so that anytime I was making a bit of a silly mistake, it
could be picked up quickly and not cause too much disruption to my work, while working
alone and in a different country, it would likely take much longer to pick up on such
mistakes.

.. rubric:: 5. What was the most challenging aspect of your project, and how did the fellowship help you overcome it?

I found it hard to wrap my head around **FEniCSx** but understood the importance of
being familiar with the foundations of FESTIM to be able to use it optimally. The
fellowship gave me the opportunity to learn about FEniCSx in a way that **wasn't just
reading their documentation** but included using my cases in FEniCSx to understand
it better.

.. rubric:: 6. Did you participate remotely or visit MIT in person? How did that work for you?

I participated in person which meant living abroad for the two-month duration of the
fellowship. This was **quite a scary thing to do, personally**, but meant that I got
to be close, and in the same time zone, as the FESTIM development team so that they
were much more accessible when I needed to ask anything. This massively increased my
efficiency of my modelling work using FESTIM v2 as **any issues could be brought up
and figured out together within the hour**.

.. rubric:: 7. Can you share a specific example of when you felt the fellowship really "clicked" for you or when you had a breakthrough?

Although a big challenge, I really appreciated the bug fix I got to help out on when
the **implicit species did not work in the discontinuous case**. It was really
interesting to see how an issue like this is approached and to learn how I could
contribute to the code by using FESTIM in the way I am, pushing the limits of its
capabilities to catch these bugs and improve the code overall.

----

Results
-------

The figures below show outputs from Tez's tungsten monoblock FESTIM v2 model.

.. TODO: Replace all three image paths below with actual simulation outputs from Tez.

.. grid:: 2
   :gutter: 3

   .. grid-item::
      :columns: 12 6 6 6

      .. figure:: /_static/images/fellows/tez_monoblock_temperature.png
         :alt: Temperature distribution in the tungsten monoblock — FESTIM v2
         :align: center
         :width: 100%

         *Temperature distribution across the tungsten monoblock.*

   .. grid-item::
      :columns: 12 6 6 6

      .. figure:: /_static/images/fellows/tez_monoblock_concentration.png
         :alt: Tritium concentration profile — FESTIM v2
         :align: center
         :width: 100%

         *Tritium concentration profile at the plasma-facing surface.*

.. figure:: /_static/images/fellows/tez_monoblock_trapping.png
   :alt: Multi-level trap occupancy — FESTIM v2
   :align: center
   :width: 65%

   *Multi-level trap occupancy distribution across the monoblock cross-section.*

----

Learning & Impact
-----------------

.. rubric:: 8. What's one advanced FESTIM feature or best practice you discovered during the fellowship that you'll use going forward?

I think that the **multi-species, multi-level occupancy trapping** capabilities coupled
with a **heat transfer problem** will be particularly useful for future work I will do
on lithium-based breeders. The ability to combine all these features into one,
relatively small script is great for making concise models without overcomplicating
the workflow. I will also be using all the **GitHub** knowledge I learnt on secondment
going forward as **version control** and having a repository has become very important
to me through the fellowship.

.. rubric:: 9. How has the fellowship changed the way you approach simulation work or your confidence with the code?

Fairly often, I am convinced that I am the problem when my simulations don't work,
thinking that I have just massively misunderstood something or made a silly mistake
somewhere. However, I've seen through the fellowship that sometimes it is just the
fact that **a feature capable of what I'm attempting hasn't yet been included or there
is a bug in the code** — and those are good things to find, not problems! It's also
made me a lot more confident in asking for help instead of staying stuck for longer
than I have to be, **a bad habit of mine**.

.. rubric:: 10. What was the outcome of your project? Did you achieve significant progress, contribute to the codebase, or accomplish something else?

The goal of the fellowship for me and UKAEA was to develop an **expert user** of the
FESTIM code who could then be a reliable point of contact for any future work that
may wish to use FESTIM, as well as between UKAEA and the PSFC team of MIT. I feel as
if this was accomplished as the **LIBRTI project** at UKAEA have already asked for
work using FESTIM to assist them since I returned from secondment. I also accomplished
much more than just a thorough understanding of FESTIM v2 as I used GitHub repositories
for the first time and learnt how to contribute to the code, a step in a good direction
for the future of my materials modelling career.

----

Broader Perspective
-------------------

.. rubric:: 11. How do you see FESTIM fitting into UKAEA's work, and what's the potential impact of your fellowship experience?

The tritium transport work happening at UKAEA will benefit from FESTIM as we look more
into **tritium inventory** taking in new breeder designs. There is interest from both
**LIBRTI** and **STEP** to use FESTIM for work along these lines as well as the
potential to use FESTIM in a **PhD** to investigate tritium transport within
lithium-based breeder pebbles. The fellowship experience has made me a reliable point
of contact for help on getting others started on using FESTIM as well as the ability
to create models where they are needed across the organisation.

.. rubric:: 12. What advice would you give to someone considering applying to the FESTIM Fellowship?

.. card::
   :class-card: sd-bg-light sd-border-2

   💬 I would tell them to **absolutely give it a shot!** It's a great way to learn
   about a really interesting modelling code while exploring the wider world of
   computational materials modelling and software development. The FESTIM team were
   all very welcoming and it was helpful to be a part of the development of not only
   the code but of all the projects that are using it.

.. rubric:: 13. Would you recommend the programme to colleagues? Why or why not?

.. pull-quote::

   "Even though it's quite nice to be the FESTIM v2 expert on site (!!!), I would
   absolutely recommend the programme to my colleagues."

   -- Tez, UKAEA

Many different modelling codes are used at UKAEA and if those who have bottlenecked
into one particular code tried out the fellowship, even in a short capacity, I think
they would realise that **there are always options when it comes to tritium transport
modelling** and to use the code which best fits their needs, rather than force the
code you like to do what you want.

----

.. seealso::

   Interested in the FESTIM Fellowship? Reach out via the
   `Support Forum <https://festim.discourse.group/>`_ or find us on
   `GitHub <https://github.com/festim-dev/FESTIM>`_.
