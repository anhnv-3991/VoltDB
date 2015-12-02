//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4-2 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2015.12.02 at 03:26:02 PM JST 
//


package org.voltdb.compiler.deploymentfile;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for systemSettingsType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="systemSettingsType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element name="temptables" minOccurs="0">
 *           &lt;complexType>
 *             &lt;complexContent>
 *               &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *                 &lt;attribute name="maxsize" type="{}memorySizeType" default="100" />
 *               &lt;/restriction>
 *             &lt;/complexContent>
 *           &lt;/complexType>
 *         &lt;/element>
 *         &lt;element name="snapshot" minOccurs="0">
 *           &lt;complexType>
 *             &lt;complexContent>
 *               &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *                 &lt;attribute name="priority" type="{}snapshotPriorityType" default="6" />
 *               &lt;/restriction>
 *             &lt;/complexContent>
 *           &lt;/complexType>
 *         &lt;/element>
 *         &lt;element name="elastic" minOccurs="0">
 *           &lt;complexType>
 *             &lt;complexContent>
 *               &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *                 &lt;attribute name="duration" type="{}elasticDurationType" default="50" />
 *                 &lt;attribute name="throughput" type="{}elasticThroughputType" default="2" />
 *               &lt;/restriction>
 *             &lt;/complexContent>
 *           &lt;/complexType>
 *         &lt;/element>
 *         &lt;element name="query" minOccurs="0">
 *           &lt;complexType>
 *             &lt;complexContent>
 *               &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *                 &lt;attribute name="timeout" type="{}latencyType" default="0" />
 *               &lt;/restriction>
 *             &lt;/complexContent>
 *           &lt;/complexType>
 *         &lt;/element>
 *       &lt;/all>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "systemSettingsType", propOrder = {

})
public class SystemSettingsType {

    protected SystemSettingsType.Temptables temptables;
    protected SystemSettingsType.Snapshot snapshot;
    protected SystemSettingsType.Elastic elastic;
    protected SystemSettingsType.Query query;

    /**
     * Gets the value of the temptables property.
     * 
     * @return
     *     possible object is
     *     {@link SystemSettingsType.Temptables }
     *     
     */
    public SystemSettingsType.Temptables getTemptables() {
        return temptables;
    }

    /**
     * Sets the value of the temptables property.
     * 
     * @param value
     *     allowed object is
     *     {@link SystemSettingsType.Temptables }
     *     
     */
    public void setTemptables(SystemSettingsType.Temptables value) {
        this.temptables = value;
    }

    /**
     * Gets the value of the snapshot property.
     * 
     * @return
     *     possible object is
     *     {@link SystemSettingsType.Snapshot }
     *     
     */
    public SystemSettingsType.Snapshot getSnapshot() {
        return snapshot;
    }

    /**
     * Sets the value of the snapshot property.
     * 
     * @param value
     *     allowed object is
     *     {@link SystemSettingsType.Snapshot }
     *     
     */
    public void setSnapshot(SystemSettingsType.Snapshot value) {
        this.snapshot = value;
    }

    /**
     * Gets the value of the elastic property.
     * 
     * @return
     *     possible object is
     *     {@link SystemSettingsType.Elastic }
     *     
     */
    public SystemSettingsType.Elastic getElastic() {
        return elastic;
    }

    /**
     * Sets the value of the elastic property.
     * 
     * @param value
     *     allowed object is
     *     {@link SystemSettingsType.Elastic }
     *     
     */
    public void setElastic(SystemSettingsType.Elastic value) {
        this.elastic = value;
    }

    /**
     * Gets the value of the query property.
     * 
     * @return
     *     possible object is
     *     {@link SystemSettingsType.Query }
     *     
     */
    public SystemSettingsType.Query getQuery() {
        return query;
    }

    /**
     * Sets the value of the query property.
     * 
     * @param value
     *     allowed object is
     *     {@link SystemSettingsType.Query }
     *     
     */
    public void setQuery(SystemSettingsType.Query value) {
        this.query = value;
    }


    /**
     * <p>Java class for anonymous complex type.
     * 
     * <p>The following schema fragment specifies the expected content contained within this class.
     * 
     * <pre>
     * &lt;complexType>
     *   &lt;complexContent>
     *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
     *       &lt;attribute name="duration" type="{}elasticDurationType" default="50" />
     *       &lt;attribute name="throughput" type="{}elasticThroughputType" default="2" />
     *     &lt;/restriction>
     *   &lt;/complexContent>
     * &lt;/complexType>
     * </pre>
     * 
     * 
     */
    @XmlAccessorType(XmlAccessType.FIELD)
    @XmlType(name = "")
    public static class Elastic {

        @XmlAttribute(name = "duration")
        protected Integer duration;
        @XmlAttribute(name = "throughput")
        protected Integer throughput;

        /**
         * Gets the value of the duration property.
         * 
         * @return
         *     possible object is
         *     {@link Integer }
         *     
         */
        public int getDuration() {
            if (duration == null) {
                return  50;
            } else {
                return duration;
            }
        }

        /**
         * Sets the value of the duration property.
         * 
         * @param value
         *     allowed object is
         *     {@link Integer }
         *     
         */
        public void setDuration(Integer value) {
            this.duration = value;
        }

        /**
         * Gets the value of the throughput property.
         * 
         * @return
         *     possible object is
         *     {@link Integer }
         *     
         */
        public int getThroughput() {
            if (throughput == null) {
                return  2;
            } else {
                return throughput;
            }
        }

        /**
         * Sets the value of the throughput property.
         * 
         * @param value
         *     allowed object is
         *     {@link Integer }
         *     
         */
        public void setThroughput(Integer value) {
            this.throughput = value;
        }

    }


    /**
     * <p>Java class for anonymous complex type.
     * 
     * <p>The following schema fragment specifies the expected content contained within this class.
     * 
     * <pre>
     * &lt;complexType>
     *   &lt;complexContent>
     *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
     *       &lt;attribute name="timeout" type="{}latencyType" default="0" />
     *     &lt;/restriction>
     *   &lt;/complexContent>
     * &lt;/complexType>
     * </pre>
     * 
     * 
     */
    @XmlAccessorType(XmlAccessType.FIELD)
    @XmlType(name = "")
    public static class Query {

        @XmlAttribute(name = "timeout")
        protected Integer timeout;

        /**
         * Gets the value of the timeout property.
         * 
         * @return
         *     possible object is
         *     {@link Integer }
         *     
         */
        public int getTimeout() {
            if (timeout == null) {
                return  0;
            } else {
                return timeout;
            }
        }

        /**
         * Sets the value of the timeout property.
         * 
         * @param value
         *     allowed object is
         *     {@link Integer }
         *     
         */
        public void setTimeout(Integer value) {
            this.timeout = value;
        }

    }


    /**
     * <p>Java class for anonymous complex type.
     * 
     * <p>The following schema fragment specifies the expected content contained within this class.
     * 
     * <pre>
     * &lt;complexType>
     *   &lt;complexContent>
     *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
     *       &lt;attribute name="priority" type="{}snapshotPriorityType" default="6" />
     *     &lt;/restriction>
     *   &lt;/complexContent>
     * &lt;/complexType>
     * </pre>
     * 
     * 
     */
    @XmlAccessorType(XmlAccessType.FIELD)
    @XmlType(name = "")
    public static class Snapshot {

        @XmlAttribute(name = "priority")
        protected Integer priority;

        /**
         * Gets the value of the priority property.
         * 
         * @return
         *     possible object is
         *     {@link Integer }
         *     
         */
        public int getPriority() {
            if (priority == null) {
                return  6;
            } else {
                return priority;
            }
        }

        /**
         * Sets the value of the priority property.
         * 
         * @param value
         *     allowed object is
         *     {@link Integer }
         *     
         */
        public void setPriority(Integer value) {
            this.priority = value;
        }

    }


    /**
     * <p>Java class for anonymous complex type.
     * 
     * <p>The following schema fragment specifies the expected content contained within this class.
     * 
     * <pre>
     * &lt;complexType>
     *   &lt;complexContent>
     *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
     *       &lt;attribute name="maxsize" type="{}memorySizeType" default="100" />
     *     &lt;/restriction>
     *   &lt;/complexContent>
     * &lt;/complexType>
     * </pre>
     * 
     * 
     */
    @XmlAccessorType(XmlAccessType.FIELD)
    @XmlType(name = "")
    public static class Temptables {

        @XmlAttribute(name = "maxsize")
        protected Integer maxsize;

        /**
         * Gets the value of the maxsize property.
         * 
         * @return
         *     possible object is
         *     {@link Integer }
         *     
         */
        public int getMaxsize() {
            if (maxsize == null) {
                return  100;
            } else {
                return maxsize;
            }
        }

        /**
         * Sets the value of the maxsize property.
         * 
         * @param value
         *     allowed object is
         *     {@link Integer }
         *     
         */
        public void setMaxsize(Integer value) {
            this.maxsize = value;
        }

    }

}
