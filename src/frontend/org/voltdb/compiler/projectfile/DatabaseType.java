//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4-2 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2015.12.02 at 03:26:02 PM JST 
//


package org.voltdb.compiler.projectfile;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for databaseType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="databaseType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;all>
 *         &lt;element name="groups" type="{}groupsType" minOccurs="0"/>
 *         &lt;element name="roles" type="{}rolesType" minOccurs="0"/>
 *         &lt;element name="schemas" type="{}schemasType"/>
 *         &lt;element name="procedures" type="{}proceduresType" minOccurs="0"/>
 *         &lt;element name="partitions" type="{}partitionsType" minOccurs="0"/>
 *         &lt;element name="classdependencies" type="{}classdependenciesType" minOccurs="0"/>
 *         &lt;element name="export" type="{}exportType" minOccurs="0"/>
 *       &lt;/all>
 *       &lt;attribute name="name" type="{http://www.w3.org/2001/XMLSchema}string" default="database" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "databaseType", propOrder = {

})
public class DatabaseType {

    protected GroupsType groups;
    protected RolesType roles;
    @XmlElement(required = true)
    protected SchemasType schemas;
    protected ProceduresType procedures;
    protected PartitionsType partitions;
    protected ClassdependenciesType classdependencies;
    protected ExportType export;
    @XmlAttribute(name = "name")
    protected String name;

    /**
     * Gets the value of the groups property.
     * 
     * @return
     *     possible object is
     *     {@link GroupsType }
     *     
     */
    public GroupsType getGroups() {
        return groups;
    }

    /**
     * Sets the value of the groups property.
     * 
     * @param value
     *     allowed object is
     *     {@link GroupsType }
     *     
     */
    public void setGroups(GroupsType value) {
        this.groups = value;
    }

    /**
     * Gets the value of the roles property.
     * 
     * @return
     *     possible object is
     *     {@link RolesType }
     *     
     */
    public RolesType getRoles() {
        return roles;
    }

    /**
     * Sets the value of the roles property.
     * 
     * @param value
     *     allowed object is
     *     {@link RolesType }
     *     
     */
    public void setRoles(RolesType value) {
        this.roles = value;
    }

    /**
     * Gets the value of the schemas property.
     * 
     * @return
     *     possible object is
     *     {@link SchemasType }
     *     
     */
    public SchemasType getSchemas() {
        return schemas;
    }

    /**
     * Sets the value of the schemas property.
     * 
     * @param value
     *     allowed object is
     *     {@link SchemasType }
     *     
     */
    public void setSchemas(SchemasType value) {
        this.schemas = value;
    }

    /**
     * Gets the value of the procedures property.
     * 
     * @return
     *     possible object is
     *     {@link ProceduresType }
     *     
     */
    public ProceduresType getProcedures() {
        return procedures;
    }

    /**
     * Sets the value of the procedures property.
     * 
     * @param value
     *     allowed object is
     *     {@link ProceduresType }
     *     
     */
    public void setProcedures(ProceduresType value) {
        this.procedures = value;
    }

    /**
     * Gets the value of the partitions property.
     * 
     * @return
     *     possible object is
     *     {@link PartitionsType }
     *     
     */
    public PartitionsType getPartitions() {
        return partitions;
    }

    /**
     * Sets the value of the partitions property.
     * 
     * @param value
     *     allowed object is
     *     {@link PartitionsType }
     *     
     */
    public void setPartitions(PartitionsType value) {
        this.partitions = value;
    }

    /**
     * Gets the value of the classdependencies property.
     * 
     * @return
     *     possible object is
     *     {@link ClassdependenciesType }
     *     
     */
    public ClassdependenciesType getClassdependencies() {
        return classdependencies;
    }

    /**
     * Sets the value of the classdependencies property.
     * 
     * @param value
     *     allowed object is
     *     {@link ClassdependenciesType }
     *     
     */
    public void setClassdependencies(ClassdependenciesType value) {
        this.classdependencies = value;
    }

    /**
     * Gets the value of the export property.
     * 
     * @return
     *     possible object is
     *     {@link ExportType }
     *     
     */
    public ExportType getExport() {
        return export;
    }

    /**
     * Sets the value of the export property.
     * 
     * @param value
     *     allowed object is
     *     {@link ExportType }
     *     
     */
    public void setExport(ExportType value) {
        this.export = value;
    }

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getName() {
        if (name == null) {
            return "database";
        } else {
            return name;
        }
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setName(String value) {
        this.name = value;
    }

}
